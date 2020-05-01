import os
import sys
import torch
import argparse
import yaml
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
from gnn.database.predictor import (
    PredictionBySmilesReaction,
    PredictionByStructLabelFeatFiles,
)
from gnn.utils import yaml_load, load_checkpoints


def parse_args():
    parser = argparse.ArgumentParser(description="BDENet bond energy predictor")

    parser.add_argument(
        "-i", "--infile", type=str, help="name of input file storing the molecules"
    )
    parser.add_argument(
        "-o", "--outfile", type=str, help="name of output file for the results"
    )
    parser.add_argument(
        "-m", "--molecule", type=str, help="smiles string of the molecule"
    )
    parser.add_argument(
        "-c", "--charge", type=int, default=0, help="charge of the molecule"
    )
    parser.add_argument(
        "-t",
        "--format",
        type=str,
        default="smi",
        choices=["smi", "sdf"],
        help="format of the molecules",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="20200422",
        help="directory name of the pre-trained model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="batch size for evaluation; larger value gives faster prediction speed "
        "but requires more memory. The final result should be the same regardless "
        "of the choice of this value. ",
    )

    args = parser.parse_args()

    # arguments compatibility check
    def check_compatibility(a1, a2):
        if getattr(args, a1) is not None and getattr(args, a2) is not None:
            raise ValueError(f"Arguments `{a1}` and `{a2}` should not be used together.")

    check_compatibility("molecule", "infile")

    return args


def get_predictor(args):

    # TODO add support for other format

    # for now, we only support file node
    # args.infile = "/Users/mjwen/Applications/db_access/prediction/smiles_reactions.csv"
    # args.outfile = "smiles_reactions_rst.csv"
    if args.infile is None:
        print("Argument `--infile` required but not provided.")
        print("To see the usage: `python predict_gated_electrolyte_rxn_ntwk.py -h`")
        sys.exit(0)

    unsupported = False

    if args.format == "smi":

        # single smiles string
        if args.molecule is not None:
            raise NotImplementedError

        # smiles csv file
        elif args.infile is not None:
            predictor = PredictionBySmilesReaction(args.infile)
            sdf_file = "/tmp/struct.sdf"
            label_file = "/tmp/label.yaml"
            feature_file = "/tmp/feature.yaml"
            predictor.convert_format(sdf_file, label_file, feature_file)

        else:
            unsupported = True

    elif args.format == "sdf":

        # sdf file
        if args.infile is not None:
            predictor = PredictionByStructLabelFeatFiles(args.infile)
            sdf_file, label_file, feature_file = predictor.convert_format()
        else:
            unsupported = True

    if unsupported:
        print("Unsupported combinatoin of arguments combination.")
        print("To see the usage: `python predict_gated_electrolyte_rxn_ntwk.py -h`")
        sys.exit(0)

    return predictor, sdf_file, label_file, feature_file


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


def get_grapher():
    atom_featurizer = AtomFeaturizer()
    bond_featurizer = BondAsNodeFeaturizer(length_featurizer=None)
    global_featurizer = GlobalFeaturizerCharge()
    grapher = HeteroMoleculeGraph(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        global_featurizer=global_featurizer,
        self_loop=True,
    )
    return grapher


def main(args):

    # convert input to model files
    predictor, sdf_file, label_file, feature_file = get_predictor(args)

    # load dataset
    model_dir = os.path.join(os.path.dirname(gnn.__file__), "pre_trained", args.model)
    dataset = ElectrolyteReactionNetworkDataset(
        grapher=get_grapher(),
        sdf_file=sdf_file,
        label_file=label_file,
        feature_file=feature_file,
        feature_transformer=True,
        label_transformer=True,
        state_dict_filename=os.path.join(model_dir, "dataset_state_dict.pkl"),
    )
    data_loader = DataLoaderReactionNetwork(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    # model
    feature_names = ["atom", "bond", "global"]
    set2set_ntypes_direct = ["global"]

    # NOTE cannot use gnn.utils.yaml_load, seems a bug in yaml.
    #  see: https://github.com/yaml/pyyaml/issues/266
    with open(os.path.join(model_dir, "train_args.yaml"), "r") as f:
        model_args = yaml.load(f, Loader=yaml.Loader)

    model = GatedGCNReactionNetwork(
        in_feats=dataset.feature_size,
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
        set2set_ntypes_direct=set2set_ntypes_direct,
        fc_num_layers=model_args.fc_num_layers,
        fc_hidden_size=model_args.fc_hidden_size,
        fc_batch_norm=model_args.fc_batch_norm,
        fc_activation=model_args.fc_activation,
        fc_dropout=model_args.fc_dropout,
        outdim=1,
        conv="GatedGCNConv",
    )
    load_checkpoints({"model": model}, filename=os.path.join(model_dir, "checkpoint.pkl"))

    # evaluate
    predictions = evaluate(model, feature_names, data_loader)

    # in case some entry fail
    if len(predictions) != len(dataset.failed):
        pred = []
        idx = 0
        for entry_id, failed in dataset.failed.items():
            if failed:
                pred.append(None)
            else:
                pred.append(predictions[idx])
                idx += 1
        predictions = pred

    # write the results
    predictor.write_results(predictions, args.outfile)


if __name__ == "__main__":
    args = parse_args()
    main(args)
