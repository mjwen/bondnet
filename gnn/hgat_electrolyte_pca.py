import numpy as np
import torch
import argparse
from gnn.model.hgat_bond import HGATBond
from gnn.data.dataset import train_validation_test_split
from gnn.data.electrolyte import ElectrolyteBondDataset
from gnn.data.dataloader import DataLoaderBond
from gnn.utils import seed_torch, load_checkpoints
from gnn.analysis.feature_analyzer import PCAAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description="HGATBond")

    # model
    parser.add_argument(
        "--num-gat-layers", type=int, default=3, help="number of GAT layers"
    )
    parser.add_argument(
        "--gat-hidden-size",
        type=int,
        nargs="+",
        default=[32, 32, 64],
        help="number of hidden units of GAT layers",
    )
    parser.add_argument(
        "--num-heads", type=int, default=1, help="number of hidden attention heads"
    )
    parser.add_argument(
        "--feat-drop", type=float, default=0.0, help="input feature dropout"
    )
    parser.add_argument("--attn-drop", type=float, default=0.0, help="attention dropout")
    parser.add_argument(
        "--negative-slope",
        type=float,
        default=0.2,
        help="the negative slope of leaky relu",
    )

    # parser.add_argument(
    #    "--residual", action="store_true", default=True, help="use residual connection"
    # )
    parser.add_argument("--residual", type=int, default=1, help="use residual connection")

    parser.add_argument(
        "--num-fc-layers", type=int, default=3, help="number of feed-forward layers"
    )
    parser.add_argument(
        "--fc-hidden-size",
        type=int,
        nargs="+",
        default=[128, 64, 32],
        help="number of hidden units of fc layers",
    )

    # training
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index. -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=100, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")

    # output file (needed by hypertunity)
    parser.add_argument(
        "--output_file", type=str, default="results.pkl", help="name of output file"
    )

    parser.add_argument("--restore", type=int, default=0, help="read checkpoints")

    args = parser.parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        args.device = torch.device("cuda:{}".format(args.gpu))
    else:
        args.device = None

    # if len(args.gat_hidden_size) == 1:
    #     args.gat_hidden_size = args.gat_hidden_size * args.num_gat_layers
    # else:
    #     assert len(args.gat_hidden_size) == args.num_gat_layers, (
    #         "length of `gat-hidden-size` should be equal to `num-gat-layers`, but got "
    #         "{} and {}.".format(args.gat_hidden_size, args.num_gat_layers)
    #     )
    #
    # if len(args.fc_hidden_size) == 1:
    #     args.fc_hidden_size = args.fc_hidden_size * args.num_fc_layers
    # else:
    #     assert len(args.fc_hidden_size) == args.num_fc_layers, (
    #     "length of `fc-hidden-size` should be equal to `num-fc-layers`, but got "
    #     "{} and {}.".format(args.fc_hidden_size, args.num_fc_layers)
    # )

    if len(args.gat_hidden_size) == 1:
        val = args.gat_hidden_size[0]
        args.gat_hidden_size = [val * 2 ** i for i in range(args.num_gat_layers)]
    else:
        assert len(args.gat_hidden_size) == args.num_gat_layers, (
            "length of `gat-hidden-size` should be equal to `num-gat-layers`, but got "
            "{} and {}.".format(args.gat_hidden_size, args.num_gat_layers)
        )

    if len(args.fc_hidden_size) == 1:
        val = args.fc_hidden_size[0]
        args.fc_hidden_size = [val // 2 ** i for i in range(args.num_fc_layers)]
    else:
        assert len(args.fc_hidden_size) == args.num_fc_layers, (
            "length of `fc-hidden-size` should be equal to `num-fc-layers`, but got "
            "{} and {}.".format(args.fc_hidden_size, args.num_fc_layers)
        )

    return args


def embedding(model, nodes, all_data_loader, text_filename, plot_filename, device=None):
    model.eval()

    all_feature = []
    all_label = []
    with torch.no_grad():
        for data_loader in all_data_loader:
            feature_data = []
            label_data = []
            for bg, label, scale in data_loader:
                feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes}
                if device is not None:
                    feats = {k: v.to(device) for k, v in feats.items()}
                feats = model.feature_before_fc(bg, feats)

                indices = [int(i) for i, v in enumerate(label["indicator"]) if v == 1]
                feature_data.append(feats[indices])
                label_data.append(label["value"][indices])

            all_feature.append(np.concatenate(feature_data))
            all_label.append(np.concatenate(label_data))

    PCAAnalyzer.embedding(all_feature, all_label, text_filename, plot_filename)


def main(args):

    ### dataset
    sdf_file = "~/Applications/mongo_db_access/extracted_data/struct_n200.sdf"
    label_file = "~/Applications/mongo_db_access/extracted_data/label_n200.txt"
    dataset = ElectrolyteBondDataset(
        sdf_file,
        label_file,
        self_loop=True,
        grapher="hetero",
        bond_length_featurizer="rbf",
    )
    trainset, valset, testset = train_validation_test_split(
        dataset, validation=0.1, test=0.1
    )
    print(
        "Trainset size: {}, valset size: {}: testset size: {}.".format(
            len(trainset), len(valset), len(testset)
        )
    )

    train_loader = DataLoaderBond(trainset, batch_size=args.batch_size, shuffle=True)
    # larger val and test set batch_size is faster but needs more memory
    # adjust the batch size of to fit memory
    bs = max(len(valset) // 10, 1)
    val_loader = DataLoaderBond(valset, batch_size=bs, shuffle=False)
    bs = max(len(testset) // 10, 1)
    test_loader = DataLoaderBond(testset, batch_size=bs, shuffle=False)

    ### model
    attn_mechanism = {
        "atom": {"edges": ["b2a", "g2a", "a2a"], "nodes": ["bond", "global", "atom"]},
        "bond": {"edges": ["a2b", "g2b", "b2b"], "nodes": ["atom", "global", "bond"]},
        "global": {"edges": ["a2g", "b2g", "g2g"], "nodes": ["atom", "bond", "global"]},
    }
    attn_order = ["atom", "bond", "global"]

    # attn_mechanism = {
    #     "atom": {"edges": ["b2a", "a2a"], "nodes": ["bond", "atom"]},
    #     "bond": {"edges": ["a2b", "b2b"], "nodes": ["atom", "bond"]},
    # }
    # attn_order = ["atom", "bond"]

    in_feats = trainset.get_feature_size(attn_order)
    model = HGATBond(
        attn_mechanism,
        attn_order,
        in_feats,
        num_gat_layers=args.num_gat_layers,
        gat_hidden_size=args.gat_hidden_size,
        num_heads=args.num_heads,
        feat_drop=args.feat_drop,
        attn_drop=args.attn_drop,
        negative_slope=args.negative_slope,
        residual=args.residual,
        num_fc_layers=args.num_fc_layers,
        fc_hidden_size=args.fc_hidden_size,
    )
    print(model)
    if args.device is not None:
        model.to(device=args.device)

    # load saved model
    checkpoints_objs = {"model": model}
    load_checkpoints(checkpoints_objs)

    # pca analysis
    # embedding(model, attn_order, [train_loader], "pca_tr.txt", "pca_tr.pdf", args.device)
    # embedding(model, attn_order, [val_loader], "pca_va.txt", "pca_va.pdf", args.device)
    embedding(
        model,
        attn_order,
        [train_loader, val_loader],
        "pca_tr_va.txt",
        "pca_tr_va.pdf",
        args.device,
    )


# do not make it main because we need to run hypertunity
seed_torch()
args = parse_args()
print(args)
main(args)
