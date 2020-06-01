import matplotlib.pyplot as plt
import os
import torch
import argparse
import yaml
import pandas as pd
import numpy as np
import gnn
from gnn.model.gated_reaction_network import GatedGCNReactionNetwork
from gnn.data.electrolyte import ElectrolyteReactionNetworkDataset
from gnn.data.dataloader import DataLoaderReactionNetwork
from gnn.data.dataset import train_validation_test_split
from gnn.data.grapher import HeteroMoleculeGraph
from gnn.data.featurizer import (
    AtomFeaturizerMinimum,
    AtomFeaturizerFull,
    BondAsNodeFeaturizerMinimum,
    BondAsNodeFeaturizerFull,
    GlobalFeaturizer,
)

from gnn.utils import load_checkpoints, seed_torch, expand_path


def parse_args():
    parser = argparse.ArgumentParser(description="BDENet bond energy predictor")

    parser.add_argument(
        "-t",
        "--analysis-type",
        type=str,
        default="error_analysis",
        help="analysis type to conduct",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="electrolyte/20200528",
        help="directory name of the pre-trained model",
    )

    args = parser.parse_args()

    return args


def get_data_loader(sdf_file, label_file, feature_file, model, batch_size=100):

    # atom_featurizer = AtomFeaturizerFull()
    # bond_featurizer = BondAsNodeFeaturizerFull(length_featurizer=None, dative=False)
    # global_featurizer = GlobalFeaturizer(allowed_charges=[0])

    atom_featurizer = AtomFeaturizerMinimum()
    bond_featurizer = BondAsNodeFeaturizerMinimum(length_featurizer=None)
    global_featurizer = GlobalFeaturizer(allowed_charges=[-1, 0, 1])

    grapher = HeteroMoleculeGraph(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        global_featurizer=global_featurizer,
        self_loop=True,
    )

    model_dir = os.path.join(
        os.path.dirname(gnn.__file__), "prediction", "pre_trained", model
    )
    dataset = ElectrolyteReactionNetworkDataset(
        grapher=grapher,
        molecules=sdf_file,
        labels=label_file,
        extra_features=feature_file,
        feature_transformer=True,
        label_transformer=True,
        state_dict_filename=os.path.join(model_dir, "dataset_state_dict.pkl"),
    )

    _, _, testset = train_validation_test_split(dataset, validation=0.1, test=0.1)
    data_loader = DataLoaderReactionNetwork(testset, batch_size=batch_size, shuffle=False)

    return data_loader


def load_model(model):

    # NOTE cannot use gnn.utils.yaml_load, seems a bug in yaml.
    #  see: https://github.com/yaml/pyyaml/issues/266
    model_dir = os.path.join(
        os.path.dirname(gnn.__file__), "prediction", "pre_trained", model
    )
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


def read_data(filename, id_col=-1):
    """Read data as dict. Keys should be specified in the first line.

    Returns:
         dict: keys specified in the first line and each column is the values.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # header
    keys = lines[0].strip("#\n").split()

    # body
    data = []
    for line in lines[1:]:
        # strip head and rear whitespaces (including ' \t\n\r\x0b\x0c')
        line = line.strip()
        # delete empty line and comments line beginning with `#'
        if not line or line[0] == "#":
            continue
        line = line.split()
        data.append([item for item in line])
    data = np.asarray(data)

    # convert to dict
    if id_col == -1:
        id_col = len(keys) - 1

    data_dict = dict()
    for i, k in enumerate(keys):
        if i == id_col:
            data_dict[k] = np.array(data[:, i], dtype="object")
        else:
            data_dict[k] = np.array(data[:, i], dtype=np.float64)

    return data_dict


def plot_prediction_vs_target(filename, plot_name="pred_vs_target.pdf"):
    """
    Plot prediction vs target as dots, to show how far they are away from y = x.

    Args:
        filename (str): file contains the data
        plot_name (str): name of the plot file
    """

    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.gca(aspect="auto")

    data = read_data(filename)
    X = data["target"]
    Y = data["prediction"]

    xy_min = min(min(X), min(Y)) - 5
    xy_max = max(max(X), max(Y)) + 5
    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)

    # plot dots
    ax.scatter(X, Y, marker="o", ec=None, alpha=0.6)

    # plot y = x
    ax.plot([xy_min, xy_max], [xy_min, xy_max], "--", color="gray", alpha=0.8)

    # label
    ax.set_xlabel("target")
    ax.set_ylabel("prediction")

    plot_name = expand_path(plot_name)
    fig.savefig(plot_name, bbox_inches="tight")


def evaluate(model, nodes, data_loader, compute_features=False):
    model.eval()

    targets = []
    predictions = []
    ids = []
    features = []

    with torch.no_grad():
        for bg, label in data_loader:
            feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes}
            tgt = label["value"]
            norm_atom = label["norm_atom"]
            norm_bond = label["norm_bond"]
            mean = label["scaler_mean"]
            stdev = label["scaler_stdev"]

            pred = model(bg, feats, label["reaction"], norm_atom, norm_bond)
            pred = pred.view(-1) * stdev + mean

            tgt = tgt * stdev + mean

            predictions.append(pred.numpy())
            targets.append(tgt.numpy())
            ids.append([rxn.id for rxn in label["reaction"]])

            if compute_features:
                feats = model.feature_before_fc(
                    bg, feats, label["reaction"], norm_atom, norm_bond
                )
                features.append(feats.numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    ids = np.concatenate(ids)
    species = ["-".join(x.split("-")[-2:]) for x in ids]
    errors = predictions - targets

    space_removed_ids = [
        s.replace(", ", "-").replace("(", "").replace(")", "") for s in ids
    ]

    if compute_features:
        features = np.concatenate(features)
    else:
        features = None

    return space_removed_ids, targets, predictions, errors, species, features


def write_features(model, nodes, data_loader, feat_filename, meta_filename):
    ids, targets, predictions, errors, species, features = evaluate(
        model, nodes, data_loader, compute_features=True
    )

    df = pd.DataFrame(features)
    df.to_csv(feat_filename, sep="\t", header=False, index=False)
    df = pd.DataFrame(
        {
            "identifier": ids,
            "target": targets,
            "prediction": predictions,
            "error": errors,
            "species": species,
        }
    )
    df.to_csv(meta_filename, sep="\t", index=False)


def error_analysis(model, nodes, data_loader, filename):
    ids, targets, predictions, errors, species, _ = evaluate(
        model, nodes, data_loader, compute_features=False
    )

    # sort by error
    ids, targets, predictions, errors, species = zip(
        *sorted(zip(ids, targets, predictions, errors, species), key=lambda x: x[3])
    )

    df = pd.DataFrame(
        {
            "identifier": ids,
            "target": targets,
            "prediction": predictions,
            "error": errors,
            "species": species,
        }
    )
    df.to_csv(expand_path(filename), sep="\t", index=False)


def main():

    seed_torch()

    args = parse_args()

    args.analysis_type = "write_feature"
    # args.analysis_type = "error_analysis"

    # get dataset
    sdf_file = "~/Applications/db_access/mol_builder/struct_rxn_ntwk_rgrn_qc.sdf"
    label_file = "~/Applications/db_access/mol_builder/label_rxn_ntwk_rgrn_qc.yaml"
    feature_file = "~/Applications/db_access/mol_builder/feature_rxn_ntwk_rgrn_qc.yaml"

    data_loader = get_data_loader(sdf_file, label_file, feature_file, model=args.model)

    # load model
    model = load_model(model=args.model)

    feature_names = ["atom", "bond", "global"]
    # feature_names = ["atom", "bond"]

    if args.analysis_type == "write_feature":
        write_features(
            model,
            feature_names,
            data_loader,
            "~/Applications/db_access/mol_builder/post_analysis/feats.tsv",
            "~/Applications/db_access/mol_builder/post_analysis/feats_metadata.tsv",
        )
    elif args.analysis_type == "error_analysis":
        fname = "~/Applications/db_access/mol_builder/post_analysis/evaluation_error.tsv"
        error_analysis(model, feature_names, data_loader, fname)
    else:
        raise ValueError(f"not supported post analysis type: {args.analysis_type}")


if __name__ == "__main__":
    main()
