import os
import torch
import yaml
import gnn
from gnn.model.gated_reaction_network import GatedGCNReactionNetwork
from gnn.data.electrolyte import ElectrolyteReactionNetworkDataset
from gnn.data.grapher import HeteroMoleculeGraph
from gnn.data.featurizer import (
    AtomFeaturizerMinimum,
    AtomFeaturizerFull,
    BondAsNodeFeaturizerMinimum,
    BondAsNodeFeaturizerFull,
    GlobalFeaturizer,
)
from gnn.core.rdmol import read_rdkit_mols_from_file
from gnn.data.utils import get_dataset_species
from gnn.utils import load_checkpoints, expand_path, check_exists


def load_model(model_name):

    model_dir = get_model_dir(model_name)

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

    return model


def load_dataset(model_name, molecules, labels, extra_features):

    model_dir = get_model_dir(model_name)
    state_dict_filename = os.path.join(model_dir, "dataset_state_dict.pkl")

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
        fname = expand_path(molecules)
        mols = read_rdkit_mols_from_file(fname)
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

    if "nrel" in model_name:
        atom_featurizer = AtomFeaturizerFull()
        bond_featurizer = BondAsNodeFeaturizerFull(length_featurizer=None, dative=False)
        global_featurizer = GlobalFeaturizer(allowed_charges=[0])

    elif "electrolyte" in model_name:
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


def get_model_dir(model_name):
    model_dir = os.path.join(
        os.path.dirname(gnn.__file__), "prediction", "pre_trained", model_name
    )

    return model_dir
