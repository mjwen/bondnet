import torch
import yaml
import bondnet
from pathlib import Path
from bondnet.model.gated_reaction_network import GatedGCNReactionNetwork
from bondnet.data.electrolyte import ElectrolyteReactionNetworkDataset
from bondnet.data.grapher import HeteroMoleculeGraph
from bondnet.data.featurizer import (
    AtomFeaturizerMinimum,
    AtomFeaturizerFull,
    BondAsNodeFeaturizerMinimum,
    BondAsNodeFeaturizerFull,
    GlobalFeaturizer,
)
from bondnet.core.rdmol import read_rdkit_mols_from_file
from bondnet.data.utils import get_dataset_species
from bondnet.utils import load_checkpoints, check_exists

kcalPerMol2eV = 0.043363

MODEL_INFO = {
    "mesd": {
        "allowed_charge": [-1, 0, 1],
        "date": ["20200611", "20200808"],  # default to the last (should be the latest)
        "unit_converter": 1.0,
    },
    "pubchem": {
        "allowed_charge": [0],
        "date": ["20200531"],
        "unit_converter": kcalPerMol2eV,
    },
}


def load_model(model_name, pretrained=True):

    model_dir = get_model_dir(model_name)

    # NOTE cannot use bondnet.utils.yaml_load, seems a bug in yaml.
    #  see: https://github.com/yaml/pyyaml/issues/266
    with open(model_dir.joinpath("train_args.yaml"), "r") as f:
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

    if pretrained:
        load_checkpoints(
            {"model": model},
            map_location=torch.device("cpu"),
            filename=model_dir.joinpath("checkpoint.pkl"),
        )

    return model


def load_dataset(model_name, molecules, labels, extra_features):

    model_dir = get_model_dir(model_name)
    state_dict_filename = model_dir.joinpath("dataset_state_dict.pkl")

    _check_species(molecules, state_dict_filename)

    dataset = ElectrolyteReactionNetworkDataset(
        grapher=_get_grapher(model_name),
        molecules=molecules,
        labels=labels,
        extra_features=extra_features,
        feature_transformer=True,
        label_transformer=True,
        state_dict_filename=state_dict_filename,
    )

    return dataset


def _check_species(molecules, state_dict_filename):

    if isinstance(molecules, str):
        check_exists(molecules)
        mols = read_rdkit_mols_from_file(molecules)
    else:
        mols = molecules

    species = get_dataset_species(mols)

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
            f"predictions for molecule containing species: {not_supported}."
        )


def _get_grapher(model_name):

    if "pubchem" in model_name:
        atom_featurizer = AtomFeaturizerFull()
        bond_featurizer = BondAsNodeFeaturizerFull(length_featurizer=None, dative=False)
        global_featurizer = GlobalFeaturizer()

    elif "mesd" in model_name:
        atom_featurizer = AtomFeaturizerMinimum()
        bond_featurizer = BondAsNodeFeaturizerMinimum(length_featurizer=None)
        global_featurizer = GlobalFeaturizer(allowed_charges=[-1, 0, 1])

    else:
        raise ValueError(f"Cannot find grapher for model {model_name}")

    grapher = HeteroMoleculeGraph(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        global_featurizer=global_featurizer,
        self_loop=True,
    )

    return grapher


def select_model(model_name):
    """
    Select the appropriate model.

    A `model_name` can be provided in two format:
    1. `dataset/data`, in this case the model will be returned directly.
    2. `dataset`, in this case the latest model corresponds to the dataset is returned.

    Args:
        model_name (str)

    Returns:
        model (str): name the model to use
        allowed_charge (list): allowed charges for molecules
        unit_converter (float): a value to multiply to convert the energy unit to eV
    """
    model_name = model_name.strip().lower()
    if "/" in model_name:
        prefix, date = model_name.split("/")
        if date not in MODEL_INFO[prefix]["date"]:
            raise ValueError(
                f"expect model date to be one of { MODEL_INFO[prefix]['date'] }, "
                f"but got {date}."
            )
    else:
        prefix = model_name
        date = MODEL_INFO[prefix]["date"][-1]

    model = "/".join([prefix, date])
    allowed_charge = MODEL_INFO[prefix]["allowed_charge"]
    unit_converter = MODEL_INFO[prefix]["unit_converter"]

    return model, allowed_charge, unit_converter


def get_model_dir(model_name):
    model_dir = Path(bondnet.__file__).parent.joinpath(
        "prediction", "pretrained", model_name
    )

    return model_dir
