import os
import torch
import yaml
import numpy as np
import gnn
from gnn.model.gated_reaction_network import GatedGCNReactionNetwork
from gnn.data.electrolyte import ElectrolyteReactionNetworkDataset
from gnn.data.dataloader import DataLoaderReactionNetwork
from gnn.data.grapher import HeteroMoleculeGraph
from gnn.data.featurizer import (
    AtomFeaturizerMinimum,
    AtomFeaturizerFull,
    BondAsNodeFeaturizerMinimum,
    BondAsNodeFeaturizerFull,
    GlobalFeaturizer,
)
from gnn.prediction.io import (
    PredictionOneReactant,
    PredictionMultiReactant,
    PredictionSDFChargeReactionFiles,
    PredictionMolGraphReactionFiles,
)
from gnn.data.utils import get_dataset_species
from gnn.utils import load_checkpoints, expand_path
from rdkit import Chem
from rdkit import RDLogger

# RDLogger.logger().setLevel(RDLogger.CRITICAL)

MODEL_INFO = {
    "electrolyte": {"allowed_charge": [-1, 0, 1], "date": ["20200422", "20200528"]},
    "nrel": {"allowed_charge": [-1, 0, 1], "date": ["20200422"]},
}


def predict_single_molecule(
    model,
    molecule,
    charge=0,
    ring_bond=False,
    write_result=False,
    figure_name="prediction.png",
    format=None,
):
    """
    Make predictions for a single molecule.

    Args:
        model (str): The pretrained model to use for making predictions. A model should
            be of the format format `dataset/date`, e.g. `electrolyte/20200528`,
            `nrel/20200528`. It is possible to provide only the `dataset` part,
            and in this case, the latest model will be used.
        molecule (str): SMILES string or InChI string.
        charge (int): charge of the molecule.
        ring_bond (bool): whether to make predictions for ring bond.
        write_result (bool): whether to write the returned sdf to stdout.
        figure_name (str): the name of the figure to be created showing the bond energy.
        format (str): format of the molecule, if not provided, will guess based on the
            file extension.

    Returns:
        str: sdf string representing the molecules and energies.
    """

    model, allowed_charge = select_model(model)

    assert (
        charge in allowed_charge
    ), f"expect charge to be one of {allowed_charge}, but got {charge}"

    if os.path.isfile(molecule):
        if format is None:
            extension = os.path.splitext(molecule)[1]
            if extension.lower() == ".sdf":
                format = "sdf"
            elif extension.lower() == ".pdb":
                format = "pdb"
            else:
                raise RuntimeError(
                    f"Expect file format `.sdf` or `.pdb`, but got {extension}"
                )
        with open(expand_path(molecule), "r") as f:
            molecule = f.read().strip()
    else:
        if format is None:
            if molecule.lower().startswith("inchi="):
                format = "inchi"
            else:
                format = "smiles"

    predictor = PredictionOneReactant(molecule, charge, format, allowed_charge, ring_bond)

    molecules, labels, extra_features = predictor.prepare_data()
    predictions = get_prediction(model, molecules, labels, extra_features)

    return predictor.write_results(predictions, figure_name, write_result)


def predict_multiple_molecules(model, molecule_file, charge_file, out_file, format):

    model, allowed_charge = select_model(model)

    predictor = PredictionMultiReactant(
        molecule_file, charge_file, format, allowed_charge, ring_bond=False
    )
    molecules, labels, extra_features = predictor.prepare_data()
    predictions = get_prediction(model, molecules, labels, extra_features)
    return predictor.write_results(predictions, out_file)


def predict_by_reactions(
    model, molecule_file, reaction_file, charge_file, out_file, format
):

    model, allowed_charge = select_model(model)

    # sdf 3 files: mol (in sdf), charge (in plain text), reaction (csv)
    if format == "sdf":
        predictor = PredictionSDFChargeReactionFiles(
            molecule_file, charge_file, reaction_file
        )

    # mol graph 2 files, mol (json or yaml), reaction (csv)
    elif format == "graph":
        predictor = PredictionMolGraphReactionFiles(molecule_file, reaction_file)

    # # internal 3 files: sdf file, label file, feature file
    # elif format == "internal":
    #     predictor = PredictionStructLabelFeatFiles(molecule_file, label_file, feat_file)

    else:
        raise ValueError(f"not supported molecule format: {format}")

    molecules, labels, extra_features = predictor.prepare_data()
    predictions = get_prediction(model, molecules, labels, extra_features)
    return predictor.write_results(predictions, out_file)


def select_model(model_str):
    model_str = model_str.strip().lower()
    if "/" in model_str:
        prefix, date = model_str.split("/")
        if date not in MODEL_INFO[prefix]["date"]:
            raise ValueError(
                f"expect model date to be one of { MODEL_INFO[prefix]['date'] }, "
                f"but got {date}."
            )
    else:
        prefix = model_str
        date = MODEL_INFO[prefix]["date"][-1]

    model = "/".join([prefix, date])
    allowed_charge = MODEL_INFO[prefix]["allowed_charge"]

    return model, allowed_charge


def evaluate(model, nodes, data_loader, device=None):
    model.eval()

    predictions = []
    with torch.no_grad():

        for it, (bg, label) in enumerate(data_loader):
            feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes}
            norm_atom = label["norm_atom"]
            norm_bond = label["norm_bond"]
            mean = label["scaler_mean"]
            stdev = label["scaler_stdev"]

            if device is not None:
                feats = {k: v.to(device) for k, v in feats.items()}
                norm_atom = norm_atom.to(device)
                norm_bond = norm_bond.to(device)

            pred = model(bg, feats, label["reaction"], norm_atom, norm_bond)
            pred = pred.view(-1)
            pred = (pred * stdev + mean).cpu().numpy()

            predictions.append(pred)

    predictions = np.concatenate(predictions)

    return predictions


def get_grapher(model):
    if "nrel" in model:
        atom_featurizer = AtomFeaturizerFull()
        bond_featurizer = BondAsNodeFeaturizerFull(length_featurizer=None, dative=False)
        global_featurizer = GlobalFeaturizer(allowed_charges=[0])
    elif "electrolyte" in model:
        atom_featurizer = AtomFeaturizerMinimum()
        bond_featurizer = BondAsNodeFeaturizerMinimum(length_featurizer=None)
        global_featurizer = GlobalFeaturizer(allowed_charges=[-1, 0, 1])
    else:
        raise Exception

    grapher = HeteroMoleculeGraph(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        global_featurizer=global_featurizer,
        self_loop=True,
    )

    return grapher


def get_prediction(model, molecules, labels, extra_features):

    model_dir = os.path.join(
        os.path.dirname(gnn.__file__), "prediction", "pre_trained", model
    )
    state_dict_filename = os.path.join(model_dir, "dataset_state_dict.pkl")

    if isinstance(molecules, str):
        mols = [m for m in Chem.SDMolSupplier(molecules, sanitize=True, removeHs=False)]
    else:
        mols = molecules
    species = get_dataset_species(mols)

    # check species are supported by dataset
    supported_species = torch.load(state_dict_filename)["species"]
    not_supported = []
    for s in species:
        if s not in supported_species:
            not_supported.append(s)
    if not_supported:
        not_supported = ",".join(not_supported)
        supported = ",".join(supported_species)
        raise ValueError(
            f"Model trained with a dataset having species: {supported}; Cannot make "
            f"predictions for molecule containing species: {not_supported}"
        )

    # load dataset
    dataset = ElectrolyteReactionNetworkDataset(
        grapher=get_grapher(model),
        molecules=molecules,
        labels=labels,
        extra_features=extra_features,
        feature_transformer=True,
        label_transformer=True,
        state_dict_filename=state_dict_filename,
    )
    data_loader = DataLoaderReactionNetwork(dataset, batch_size=100, shuffle=False)

    # model
    feature_names = ["atom", "bond", "global"]
    # feature_names = ["atom", "bond"]

    # NOTE cannot use gnn.utils.yaml_load, seems a bug in yaml.
    #  see: https://github.com/yaml/pyyaml/issues/266
    with open(os.path.join(model_dir, "train_args.yaml"), "r") as f:
        model_args = yaml.load(f, Loader=yaml.Loader)

    model = GatedGCNReactionNetwork(
        in_feats=model_args.feature_size,
        embedding_size=model_args.embedding_size,
        gated_num_layers=model_args.gated_num_layers,
        gated_hidden_size=model_args.gated_hidden_size,
        gated_num_fc_layers=model_args.gated_num_fc_layers,
        gated_graph_norm=model_args.gated_graph_norm,
        gated_batch_norm=model_args.gated_batch_norm,
        gated_activation=model_args.gated_activation,
        gated_residual=model_args.gated_residual,
        gated_dropout=model_args.gated_dropout,
        num_lstm_iters=model_args.num_lstm_iters,
        num_lstm_layers=model_args.num_lstm_layers,
        set2set_ntypes_direct=model_args.set2set_ntypes_direct,
        fc_num_layers=model_args.fc_num_layers,
        fc_hidden_size=model_args.fc_hidden_size,
        fc_batch_norm=model_args.fc_batch_norm,
        fc_activation=model_args.fc_activation,
        fc_dropout=model_args.fc_dropout,
        outdim=1,
        conv="GatedGCNConv",
    )
    load_checkpoints(
        {"model": model},
        map_location=torch.device("cpu"),
        filename=os.path.join(model_dir, "checkpoint.pkl"),
    )

    # evaluate
    predictions = evaluate(model, feature_names, data_loader)

    # in case some entry fail
    if len(predictions) != len(dataset.failed):
        pred = []
        idx = 0
        for failed in dataset.failed:
            if failed:
                pred.append(None)
            else:
                pred.append(predictions[idx])
                idx += 1
        predictions = pred

    return predictions
