from gnn.utils import expand_path
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
from gnn.data.grapher import HeteroMoleculeGraph
from gnn.data.featurizer import (
    AtomFeaturizer,
    BondAsNodeFeaturizer,
    GlobalFeaturizerCharge,
)
from gnn.utils import load_checkpoints


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
        default="20200422",
        help="directory name of the pre-trained model",
    )

    args = parser.parse_args()

    return args


def get_data_loader(sdf_file, label_file, feature_file, batch_size=100, model="20200422"):

    atom_featurizer = AtomFeaturizer()
    bond_featurizer = BondAsNodeFeaturizer(length_featurizer=None)
    global_featurizer = GlobalFeaturizerCharge()
    grapher = HeteroMoleculeGraph(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        global_featurizer=global_featurizer,
        self_loop=True,
    )

    model_dir = os.path.join(os.path.dirname(gnn.__file__), "pre_trained", model)
    dataset = ElectrolyteReactionNetworkDataset(
        grapher=grapher,
        sdf_file=sdf_file,
        label_file=label_file,
        feature_file=feature_file,
        feature_transformer=True,
        label_transformer=True,
        state_dict_filename=os.path.join(model_dir, "dataset_state_dict.pkl"),
    )
    data_loader = DataLoaderReactionNetwork(dataset, batch_size=batch_size, shuffle=False)

    return data_loader


def load_model(model="20200422"):

    # NOTE cannot use gnn.utils.yaml_load, seems a bug in yaml.
    #  see: https://github.com/yaml/pyyaml/issues/266
    model_dir = os.path.join(os.path.dirname(gnn.__file__), "pre_trained", model)
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
    load_checkpoints({"model": model}, filename=os.path.join(model_dir, "checkpoint.pkl"))

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


def write_features(model, nodes, data_loader, feat_filename, meta_filename):
    model.eval()

    feature_data = []
    label_data = []
    ids = []

    with torch.no_grad():

        for bg, label in data_loader:
            feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes}
            norm_atom = label["norm_atom"]
            norm_bond = label["norm_bond"]

            feats = model.feature_before_fc(
                bg, feats, label["reaction"], norm_atom, norm_bond
            )

            feature_data.append(feats.cpu().numpy())

            target = (
                torch.mul(label["value"], label["scaler_stdev"]) + label["scaler_mean"]
            )
            label_data.append(target.numpy())
            ids.append([rxn.id for rxn in label["reaction"]])

    feature_data = np.concatenate(feature_data)
    label_data = np.concatenate(label_data)
    ids = np.concatenate(ids)

    # write files
    df = pd.DataFrame(feature_data)
    df.to_csv(feat_filename, sep="\t", header=False, index=False)
    df = pd.DataFrame({"ids": ids, "energy": label_data})
    df.to_csv(meta_filename, sep="\t", index=False)


def error_analysis(model, nodes, data_loader, filename):
    model.eval()

    predictions = []
    targets = []
    ids = []

    with torch.no_grad():

        for bg, label in data_loader:
            feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes}
            tgt = label["value"]
            mean = label["scaler_mean"]
            stdev = label["scaler_stdev"]
            norm_atom = label["norm_atom"]
            norm_bond = label["norm_bond"]

            pred = model(bg, feats, label["reaction"], norm_atom, norm_bond)
            pred = pred.view(-1)

            pred = pred.cpu() * stdev + mean
            tgt = tgt * stdev + mean
            predictions.append(pred.numpy())
            targets.append(tgt.numpy())
            ids.append([rxn.id for rxn in label["reaction"]])

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    ids = np.concatenate(ids)

    write_error(predictions, targets, ids, sort=True, filename=filename)


def write_error(predictions, targets, ids, sort=True, filename="error.txt"):
    """
    Write the error to file.

    Args:
        predictions (list): model prediction.
        targets (list): reference value.
        ids (list): ids associated with errors.
        sort (bool): whether to sort the error from low to high.
        filename (str): filename to write out the result.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    errors = predictions - targets

    if sort:
        errors, predictions, targets, ids = zip(
            *sorted(zip(errors, predictions, targets, ids), key=lambda pair: pair[0])
        )
    with open(expand_path(filename), "w") as f:
        f.write("# error    prediction    target    id\n")
        for e, p, t, i in zip(errors, predictions, targets, ids):
            f.write("{:13.5e} {:13.5e} {:13.5e}    {}\n".format(e, p, t, i))

        # MAE, RMSE, and MAX Error
        abs_e = np.abs(errors)
        mae = np.mean(abs_e)
        rmse = np.sqrt(np.mean(np.square(errors)))
        max_e_idx = np.argmax(abs_e)

        f.write("\n")
        f.write(f"# MAE: {mae}\n")
        f.write(f"# RMSE: {rmse}\n")
        f.write(f"# MAX error: {abs_e[max_e_idx]}   {ids[max_e_idx]}\n")


def main():

    args = parse_args()

    args.analysis_type = "write_feature"
    # args.analysis_type = "error_analysis"

    # get dataset
    sdf_file = "~/Applications/db_access/mol_builder/struct_rxn_ntwk_rgrn_n200.sdf"
    label_file = "~/Applications/db_access/mol_builder/label_rxn_ntwk_rgrn_n200.yaml"
    feature_file = "~/Applications/db_access/mol_builder/feature_rxn_ntwk_rgrn_n200.yaml"

    data_loader = get_data_loader(sdf_file, label_file, feature_file, model=args.model)

    # load model
    model = load_model(model=args.model)

    feature_names = ["atom", "bond", "global"]
    # feature_names = ["atom", "bond"]

    if args.analysis_type == "write_feature":
        write_features(
            model, feature_names, data_loader, "feats.tsv", "feats_metadata.tsv",
        )
    elif args.analysis_type == "error_analysis":
        fname = "evaluation_error.txt"
        error_analysis(model, feature_names, data_loader, fname)
    else:
        raise ValueError(f"not supported post analysis type: {args.analysis_type}")


if __name__ == "__main__":
    main()
