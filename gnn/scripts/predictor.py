import os
import torch
import yaml
import click
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
from gnn.prediction.prediction import (
    PredictionOneReactant,
    PredictionMultiReactant,
    PredictionSDFChargeReactionFiles,
    PredictionMolGraphReactionFiles,
)
from gnn.data.utils import get_dataset_species
from gnn.utils import load_checkpoints
from rdkit import Chem

# RDLogger.logger().setLevel(RDLogger.CRITICAL)

LATEST_NREL_MODEL = "20200422"
LATEST_ELECTROLYTE_MODEL = "20200528"

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS, invoke_without_command=False)
@click.option(
    "-m",
    "--model",
    type=click.Choice(["nrel", "electrolyte"], case_sensitive=False),
    default="electrolyte",
    show_default=True,
    help="prediction using model trained to the dataset",
)
@click.version_option(version=gnn.__version__)
@click.pass_context
def cli(ctx, model):
    model = model.lower()
    if model == "nrel":
        model = os.path.join(model, LATEST_NREL_MODEL)
        allowed_charge = [0]
    else:
        model = os.path.join(model, LATEST_ELECTROLYTE_MODEL)
        allowed_charge = [-1, 0, 1]

    model_info = {"model": model, "allowed_charge": allowed_charge}

    ctx.obj = model_info


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("molecule", type=str)
@click.option("--charge", type=int, default=0, help="charge of the molecule")
@click.pass_obj
def single(model_info, molecule, charge):
    """
    Make predictions for a molecule.

    MOLECULE is a SMILES string or InChI string.
    """

    if molecule.lower().startswith("inchi="):
        format = "inchi"
    else:
        format = "smiles"
    predictor = PredictionOneReactant(
        molecule, charge, format, model_info["allowed_charge"], ring_bond=False
    )
    molecules, labels, extra_features = predictor.prepare_data()
    predictions = get_prediction(model_info["model"], molecules, labels, extra_features)
    predictor.write_results(predictions)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("molecule-file", type=click.Path(exists=True))
@click.option(
    "-c",
    "--charge-file",
    type=click.Path(exists=True),
    help="charges of molecules; if not provided, charges set to 0",
)
@click.option(
    "-o",
    "--out-file",
    type=click.Path(exists=False),
    help="file to write results; if not provided, write to stdout",
)
@click.option(
    "-t",
    "--format",
    type=click.Choice(["sdf", "pdb", "smiles", "inchi"], case_sensitive=False),
    default="sdf",
    show_default=True,
    help="format of molecules",
)
@click.pass_obj
def multiple(model_info, molecule_file, charge_file, out_file, format):
    """
    Make predictions for multiple molecules.
    """

    predictor = PredictionMultiReactant(
        molecule_file, charge_file, format, model_info["allowed_charge"], ring_bond=False
    )
    molecules, labels, extra_features = predictor.prepare_data()
    predictions = get_prediction(model_info["model"], molecules, labels, extra_features)
    predictor.write_results(predictions, out_file)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("molecule-file", type=click.Path(exists=True))
@click.argument("reaction-file", type=click.Path(exists=True))
@click.option(
    "-c",
    "--charge-file",
    type=click.Path(exists=True),
    help="charges of molecules; if not provided, charges set to 0",
)
@click.option(
    "-o",
    "--out-file",
    type=click.Path(exists=False),
    help="file to write results; if not provided, write to stdout",
)
@click.option(
    "-t",
    "--format",
    type=click.Choice(["sdf", "graph", "pdb", "smiles", "inchi"], case_sensitive=False),
    default="sdf",
    show_default=True,
    help="format of molecules",
)
@click.pass_obj
def reaction(model_info, molecule_file, reaction_file, charge_file, out_file, format):
    """
    Make predictions for bonds given as reactions.

    MOLECULE_FILE lists all the molecules.

    REACTION_FILE is a csv file lists bond breaking reactions the molecules can form,
    specified by the index of the molecules.
    """

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
    predictions = get_prediction(model_info["model"], molecules, labels, extra_features)
    predictor.write_results(predictions, out_file)


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


if __name__ == "__main__":
    cli()
