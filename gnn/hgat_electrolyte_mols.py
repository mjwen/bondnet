import sys
import time
from datetime import datetime
import warnings
import torch
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss
from gnn.metric import WeightedL1Loss, EarlyStopping
from gnn.model.hgatmol import HGATMol
from gnn.data.dataset import train_validation_test_split
from gnn.data.electrolyte import ElectrolyteMoleculeDataset
from gnn.data.dataloader import DataLoaderMolecule
from gnn.data.grapher import HeteroMoleculeGraph
from gnn.data.featurizer import (
    AtomFeaturizerWithReactionInfo,
    BondAsNodeFeaturizer,
    GlobalFeaturizer,
)
from gnn.utils import pickle_dump, seed_torch, load_checkpoints


def parse_args():
    parser = argparse.ArgumentParser(description="HGAT")

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
        "--num-lstm-iters",
        type=int,
        default=6,
        help="number of iterations for the LSTM in set2set readout layer",
    )
    parser.add_argument(
        "--num-lstm-layers",
        type=int,
        default=3,
        help="number of layers for the LSTM in set2set readout layer",
    )

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

    if len(args.gat_hidden_size) == 1:
        args.gat_hidden_size = args.gat_hidden_size * args.num_gat_layers
    else:
        assert len(args.gat_hidden_size) == args.num_gat_layers, (
            "length of `gat-hidden-size` should be equal to `num-gat-layers`, but got "
            "{} and {}.".format(args.gat_hidden_size, args.num_gat_layers)
        )

    if len(args.fc_hidden_size) == 1:
        args.fc_hidden_size = args.fc_hidden_size * args.num_fc_layers
    else:
        assert len(args.fc_hidden_size) == args.num_fc_layers, (
            "length of `fc-hidden-size` should be equal to `num-fc-layers`, but got "
            "{} and {}.".format(args.fc_hidden_size, args.num_fc_layers)
        )

    # if len(args.gat_hidden_size) == 1:
    #     val = args.gat_hidden_size[0]
    #     args.gat_hidden_size = [val * 2 ** i for i in range(args.num_gat_layers)]
    # else:
    #     assert len(args.gat_hidden_size) == args.num_gat_layers, (
    #         "length of `gat-hidden-size` should be equal to `num-gat-layers`, but got "
    #         "{} and {}.".format(args.gat_hidden_size, args.num_gat_layers)
    #     )

    # if len(args.fc_hidden_size) == 1:
    #     val = args.fc_hidden_size[0]
    #     args.fc_hidden_size = [val // 2 ** i for i in range(args.num_fc_layers)]
    # else:
    #     assert len(args.fc_hidden_size) == args.num_fc_layers, (
    #         "length of `fc-hidden-size` should be equal to `num-fc-layers`, but got "
    #         "{} and {}.".format(args.fc_hidden_size, args.num_fc_layers)
    #     )

    return args


def train(optimizer, model, nodes, data_loader, loss_fn, metric_fn, device=None):
    """
    Args:
        metric_fn (function): the function should be using a `sum` reduction method.
    """

    model.train()

    epoch_loss = 0.0
    accuracy = 0.0
    count = 0.0

    for it, (bg, label, scale) in enumerate(data_loader):
        feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes}
        if device is not None:
            feats = {k: v.to(device=device) for k, v in feats.items()}
            label = label.to(device=device)
            if scale is not None:
                scale = scale.to(device=device)

        pred = model(bg, feats)
        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()
        accuracy += metric_fn(pred, label, scale).detach().item()
        count += len(label)

    epoch_loss /= it + 1
    accuracy /= count

    return epoch_loss, accuracy


def evaluate(model, nodes, data_loader, metric_fn, device=None):
    """
    Evaluate the accuracy of an validation set of test set.

    Args:
        metric_fn (function): the function should be using a `sum` reduction method.
    """
    model.eval()

    with torch.no_grad():
        accuracy = 0.0
        count = 0.0

        for bg, label, scale in data_loader:
            feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes}
            if device is not None:
                feats = {k: v.to(device) for k, v in feats.items()}
                label = label.to(device=device)
                if scale is not None:
                    scale = scale.to(device=device)

            pred = model(bg, feats)
            accuracy += metric_fn(pred, label, scale).detach().item()
            count += len(label)

    return accuracy / count


def get_grapher():
    atom_featurizer = AtomFeaturizerWithReactionInfo()
    bond_featurizer = BondAsNodeFeaturizer(length_featurizer="bin")
    global_featurizer = GlobalFeaturizer()
    grapher = HeteroMoleculeGraph(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        global_featurizer=global_featurizer,
        self_loop=True,
    )
    return grapher


def main(args):
    print("\n\nStart training at:", datetime.now())

    ### dataset
    sdf_file = "~/Applications/db_access/mol_builder/struct_mols_n200.sdf"
    label_file = "~/Applications/db_access/mol_builder/label_mols_n200.csv"
    feature_file = "~/Applications/db_access/mol_builder/feature_mols_n200.yaml"

    dataset = ElectrolyteMoleculeDataset(
        grapher=get_grapher(),
        sdf_file=sdf_file,
        label_file=label_file,
        feature_file=feature_file,
        properties=["atomization_energy"],
        unit_conversion=True,
    )
    print(dataset)

    trainset, valset, testset = train_validation_test_split(
        dataset, validation=0.1, test=0.1
    )
    print(
        "Trainset size: {}, valset size: {}: testset size: {}.".format(
            len(trainset), len(valset), len(testset)
        )
    )

    # we use the QM9 dataloader, they as the same type of data
    train_loader = DataLoaderMolecule(
        trainset, hetero=True, batch_size=args.batch_size, shuffle=True
    )
    # larger val and test set batch_size is faster but needs more memory
    # adjust the batch size of to fit memory
    bs = len(valset) // 10
    val_loader = DataLoaderMolecule(valset, hetero=True, batch_size=bs, shuffle=False)
    bs = len(testset) // 10
    test_loader = DataLoaderMolecule(testset, hetero=True, batch_size=bs, shuffle=False)

    ### model
    attn_mechanism = {
        "atom": {"edges": ["b2a", "g2a", "a2a"], "nodes": ["bond", "global", "atom"]},
        "bond": {"edges": ["a2b", "g2b", "b2b"], "nodes": ["atom", "global", "bond"]},
        "global": {"edges": ["a2g", "b2g", "g2g"], "nodes": ["atom", "bond", "global"]},
    }
    attn_order = ["atom", "bond", "global"]
    set2set_ntypes_direct = ["global"]

    # attn_mechanism = {
    #     "atom": {"edges": ["b2a", "a2a"], "nodes": ["bond", "atom"]},
    #     "bond": {"edges": ["a2b", "b2b"], "nodes": ["atom", "bond"]},
    # }
    # attn_order = ["atom", "bond"]
    # set2set_ntypes_direct = None

    in_feats = trainset.get_feature_size(attn_order)
    model = HGATMol(
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
        num_lstm_iters=args.num_lstm_iters,
        num_lstm_layers=args.num_lstm_layers,
        set2set_ntypes_direct=set2set_ntypes_direct,
    )
    print(model)
    if args.device is not None:
        model.to(device=args.device)

    ### optimizer, loss, and metric
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_func = MSELoss(reduction="mean")
    metric = WeightedL1Loss(reduction="sum")

    ### learning rate scheduler and stopper
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.4, patience=50, verbose=True
    )
    stopper = EarlyStopping(patience=150)

    checkpoints_objs = {"model": model, "optimizer": optimizer, "scheduler": scheduler}
    if args.restore:
        try:
            load_checkpoints(checkpoints_objs)
            print("Successfully load checkpoints")
        except FileNotFoundError as e:
            warnings.warn(str(e) + " Continue without loading checkpoints.")
            pass

    print("\n\n# Epoch     Loss         TrainAcc        ValAcc     Time (s)")
    t0 = time.time()

    for epoch in range(args.epochs):
        ti = time.time()

        # train and evaluate accuracy
        loss, train_acc = train(
            optimizer, model, attn_order, train_loader, loss_func, metric, args.device
        )
        val_acc = evaluate(model, attn_order, val_loader, metric, args.device)

        if stopper.step(val_acc, checkpoints_objs, msg="epoch " + str(epoch)):
            # save results for hyperparam tune
            pickle_dump(float(stopper.best_score), args.output_file)
            break

        scheduler.step(val_acc)

        tt = time.time() - ti

        print(
            "{:5d}   {:12.6e}   {:12.6e}   {:12.6e}   {:.2f}".format(
                epoch, loss, train_acc, val_acc, tt
            )
        )
        if epoch % 10 == 0:
            sys.stdout.flush()

    # save results for hyperparam tune
    pickle_dump(float(stopper.best_score), args.output_file)

    # load best to calculate test accuracy
    load_checkpoints(checkpoints_objs)
    test_acc = evaluate(model, attn_order, test_loader, metric, args.device)

    tt = time.time() - t0
    print("\n#TestAcc: {:12.6e} | Total time (s): {:.2f}\n".format(test_acc, tt))
    print("\nFinish training at:", datetime.now())


# do not make it main because we need to run hypertunity
seed_torch()
args = parse_args()
print(args)
main(args)
