import os
import torch
import argparse
import yaml
import numpy as np
import gnn
from gnn.model.gated_reaction_network import GatedGCNReactionNetwork
from gnn.data.electrolyte import ElectrolyteReactionNetworkDataset
from gnn.data.dataloader import DataLoaderReactionNetwork
from gnn.data.grapher import HeteroMoleculeGraph
from gnn.data.featurizer import AtomFeaturizer, BondAsNodeFeaturizer, MolWeightFeaturizer
from gnn.utils import load_checkpoints


def parse_args():
    parser = argparse.ArgumentParser(description="BDENet bond energy predictor")

    parser.add_argument(
        "-m", "--molecule", type=str, help="smiles string of the molecule",
    )
    parser.add_argument(
        "-c", "--charge", type=int, default=0, help="charge of the molecule",
    )
    parser.add_argument(
        "-f", "--file", type=str, help="name of files storing the molecules",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="smi",
        choices=["smi", "sdf"],
        help="directory name of the pre-trained model",
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
        help="batch size for evaluation; larger value gives faster prediction speed but "
        "requires more memory. The final result should be exactly the same regardless "
        "of the choice of this value. ",
    )

    args = parser.parse_args()

    # arguments compatibility check
    def check_compatibility(a1, a2):
        if getattr(args, a1) is not None and getattr(args, a2) is not None:
            raise ValueError(f"Arguments `{a1}` and `{a2}` should not be used together.")

    check_compatibility("molecule", "file")

    return args


def create_files(args):

    ### dataset
    # sdf_file = "~/Applications/db_access/mol_builder/struct_rxn_ntwk_rgrn_n200.sdf"
    # label_file = "~/Applications/db_access/mol_builder/label_rxn_ntwk_rgrn_n200.yaml"
    # feature_file = "~/Applications/db_access/mol_builder/feature_rxn_ntwk_rgrn_n200.yaml"
    sdf_file = "~/Applications/db_access/zinc_bde/zinc_struct_rxn_ntwk_rgrn_n200.sdf"
    label_file = "~/Applications/db_access/zinc_bde/zinc_label_rxn_ntwk_rgrn_n200.yaml"
    feature_file = (
        "~/Applications/db_access/zinc_bde/zinc_feature_rxn_ntwk_rgrn_n200.yaml"
    )

    return sdf_file, label_file, feature_file


def write_result(predictions, args):
    for i, x in enumerate(predictions):
        print(i, x)


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
    global_featurizer = MolWeightFeaturizer()
    grapher = HeteroMoleculeGraph(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        global_featurizer=global_featurizer,
        self_loop=True,
    )
    return grapher


def main(args):

    model_dir = os.path.join(os.path.dirname(gnn.__file__), "pre_trained", args.model)

    # convert input to model files
    sdf_file, label_file, feature_file = create_files(args)

    # load dataset
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
    load_checkpoints({"model": model}, os.path.join(model_dir, "checkpoint.pkl"))

    # evaluate
    predictions = evaluate(model, feature_names, data_loader)

    write_result(predictions, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
