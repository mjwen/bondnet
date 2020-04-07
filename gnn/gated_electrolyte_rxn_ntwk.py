import sys
import time
import warnings
import torch
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss
from gnn.metric import WeightedL1Loss, EarlyStopping
from gnn.model.gated_reaction_network import GatedGCNReactionNetwork
from gnn.data.dataset import train_validation_test_split
from gnn.data.electrolyte import ElectrolyteReactionNetworkDataset
from gnn.data.dataloader import DataLoaderReactionNetwork
from gnn.data.grapher import HeteroMoleculeGraph
from gnn.data.featurizer import AtomFeaturizer, BondAsNodeFeaturizer, MolWeightFeaturizer
from gnn.post_analysis import write_error
from gnn.utils import pickle_dump, seed_torch, load_checkpoints


def parse_args():
    parser = argparse.ArgumentParser(description="HGATReaction")

    # embedding layer
    parser.add_argument("--embedding-size", type=int, default=24)

    # gated layer
    parser.add_argument("--gated-num-layers", type=int, default=3)
    parser.add_argument("--gated-hidden-size", type=int, nargs="+", default=[64, 64, 64])
    parser.add_argument("--gated-graph-norm", type=int, default=1)
    parser.add_argument("--gated-batch-norm", type=int, default=1)
    parser.add_argument("--gated-activation", type=str, default="ReLU")
    parser.add_argument("--gated-residual", type=int, default=1)
    parser.add_argument("--gated-dropout", type=float, default="0.0")

    # readout layer
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

    # fc layer
    parser.add_argument("--fc-num-layers", type=int, default=2)
    parser.add_argument("--fc-hidden-size", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--fc-batch-norm", type=int, default=0)
    parser.add_argument("--fc-activation", type=str, default="ReLU")
    parser.add_argument("--fc-dropout", type=float, default=0.0)

    # training
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index. -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=100, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--restore", type=int, default=0, help="read checkpoints")

    # output file (needed by hypertunity)
    parser.add_argument("--output_file", type=str, default="results.pkl")

    parser.add_argument(
        "--post-analysis", type=str, default="none", help="post analysis type"
    )

    args = parser.parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        args.device = torch.device("cuda:{}".format(args.gpu))
    else:
        args.device = None

    if len(args.gated_hidden_size) == 1:
        args.gated_hidden_size = args.gated_hidden_size * args.gated_num_layers
    else:
        assert len(args.gated_hidden_size) == args.gated_num_layers, (
            "length of `gat-hidden-size` should be equal to `num-gat-layers`, but got "
            "{} and {}.".format(args.gated_hidden_size, args.gated_num_layers)
        )

    # if len(args.fc_hidden_size) == 1:
    #     args.fc_hidden_size = args.fc_hidden_size * args.num_fc_layers
    # else:
    #     assert len(args.fc_hidden_size) == args.num_fc_layers, (
    #         "length of `fc-hidden-size` should be equal to `num-fc-layers`, but got "
    #         "{} and {}.".format(args.fc_hidden_size, args.num_fc_layers)
    #     )

    # if len(args.gated_hidden_size) == 1:
    #    val = args.gated_hidden_size[0]
    #    args.gated_hidden_size = [val * 2 ** i for i in range(args.gated_num_layers)]
    # else:
    #    assert len(args.gated_hidden_size) == args.gated_num_layers, (
    #        "length of `gat-hidden-size` should be equal to `num-gat-layers`, but got "
    #        "{} and {}.".format(args.gated_hidden_size, args.gated_num_layers)
    #    )

    if len(args.fc_hidden_size) == 1:
        # val = args.fc_hidden_size[0]
        val = args.gated_hidden_size[-1]
        args.fc_hidden_size = [val // 2 ** i for i in range(args.fc_num_layers)]
    else:
        assert len(args.fc_hidden_size) == args.fc_num_layers, (
            "length of `fc-hidden-size` should be equal to `num-fc-layers`, but got "
            "{} and {}.".format(args.fc_hidden_size, args.fc_num_layers)
        )

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

    for it, (bg, label) in enumerate(data_loader):
        feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes}
        target = label["value"]
        norm_atom = label["norm_atom"]
        norm_bond = label["norm_bond"]
        stdev = label["scaler_stdev"]

        if device is not None:
            feats = {k: v.to(device) for k, v in feats.items()}
            target = target.to(device)
            norm_atom = norm_atom.to(device)
            norm_bond = norm_bond.to(device)
            stdev = stdev.to(device)

        pred = model(bg, feats, label["reaction"], norm_atom, norm_bond)
        pred = pred.view(-1)

        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()
        accuracy += metric_fn(pred, target, stdev).detach().item()
        count += len(target)

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

        for bg, label in data_loader:
            feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes}
            target = label["value"]
            norm_atom = label["norm_atom"]
            norm_bond = label["norm_bond"]
            stdev = label["scaler_stdev"]

            if device is not None:
                feats = {k: v.to(device) for k, v in feats.items()}
                target = target.to(device)
                norm_atom = norm_atom.to(device)
                norm_bond = norm_bond.to(device)
                stdev = stdev.to(device)

            pred = model(bg, feats, label["reaction"], norm_atom, norm_bond)
            pred = pred.view(-1)

            accuracy += metric_fn(pred, target, stdev).detach().item()
            count += len(target)

    return accuracy / count


def write_features(
    model, nodes, all_data_loader, feat_filename, meta_filename, device=None
):
    model.eval()

    all_feature = []
    all_label = []
    all_ids = []
    loader_names = []

    with torch.no_grad():
        for name, data_loader in all_data_loader.items():

            feature_data = []
            label_data = []
            ids = []
            for bg, label in data_loader:
                feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes}
                norm_atom = label["norm_atom"]
                norm_bond = label["norm_bond"]

                if device is not None:
                    feats = {k: v.to(device) for k, v in feats.items()}
                    norm_atom = norm_atom.to(device)
                    norm_bond = norm_bond.to(device)

                feats = model.feature_before_fc(
                    bg, feats, label["reaction"], norm_atom, norm_bond
                )

                feature_data.append(feats)

                target = (
                    torch.mul(label["value"], label["scaler_stdev"])
                    + label["scaler_mean"]
                )
                label_data.append(target.numpy())
                ids.append([rxn.id for rxn in label["reaction"]])

            all_feature.append(np.concatenate(feature_data))
            all_label.append(np.concatenate(label_data))
            all_ids.append(np.concatenate(ids))
            loader_names.append(name)

    # features
    feats = np.concatenate(all_feature)

    # metadata
    loader_source = [[nm] * len(lb) for nm, lb in zip(loader_names, all_label)]
    metadata = {
        "ids": np.concatenate(all_ids),
        "energy": np.concatenate(all_label),
        "loader": np.concatenate(loader_source),
    }

    # write files
    df = pd.DataFrame(feats)
    df.to_csv(feat_filename, sep="\t", header=False, index=False)
    df = pd.DataFrame(metadata)
    df.to_csv(meta_filename, sep="\t", index=False)


def error_analysis(model, nodes, data_loader, filename, device=None):
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

            if device is not None:
                feats = {k: v.to(device) for k, v in feats.items()}
                norm_atom = norm_atom.to(device)
                norm_bond = norm_bond.to(device)

            pred = model(bg, feats, label["reaction"], norm_atom, norm_bond)
            pred = pred.view(-1)

            pred = pred.numpy() * stdev + mean
            tgt = tgt * stdev + mean
            predictions.append(pred.numpy())
            targets.append(tgt.numpy())
            ids.append([rxn.id for rxn in label["reaction"]])

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    ids = np.concatenate(ids)

    write_error(predictions, targets, ids, sort=True, filename=filename)


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
    print("\n\nStart training at:", datetime.now())

    ### dataset
    # sdf_file = "~/Applications/db_access/mol_builder/struct_rxn_ntwk_rgrn_n200.sdf"
    # label_file = "~/Applications/db_access/mol_builder/label_rxn_ntwk_rgrn_n200.yaml"
    # feature_file = "~/Applications/db_access/mol_builder/feature_rxn_ntwk_rgrn_n200.yaml"
    sdf_file = "~/Applications/db_access/mol_builder/zinc_struct_rxn_ntwk_rgrn_n200.sdf"
    label_file = "~/Applications/db_access/mol_builder/zinc_label_rxn_ntwk_rgrn_n200.yaml"
    feature_file = (
        "~/Applications/db_access/mol_builder/zinc_feature_rxn_ntwk_rgrn_n200.yaml"
    )

    dataset = ElectrolyteReactionNetworkDataset(
        grapher=get_grapher(),
        sdf_file=sdf_file,
        label_file=label_file,
        feature_file=feature_file,
        feature_transformer=True,
        label_transformer=True,
    )

    trainset, valset, testset = train_validation_test_split(
        dataset, validation=0.1, test=0.1
    )
    print(
        "Trainset size: {}, valset size: {}: testset size: {}.".format(
            len(trainset), len(valset), len(testset)
        )
    )

    train_loader = DataLoaderReactionNetwork(
        trainset, batch_size=args.batch_size, shuffle=True
    )
    # larger val and test set batch_size is faster but needs more memory
    # adjust the batch size of to fit memory
    bs = max(len(valset) // 10, 1)
    val_loader = DataLoaderReactionNetwork(valset, batch_size=bs, shuffle=False)
    bs = max(len(testset) // 10, 1)
    test_loader = DataLoaderReactionNetwork(testset, batch_size=bs, shuffle=False)

    ### model

    feature_names = ["atom", "bond", "global"]
    set2set_ntypes_direct = ["global"]

    # feature_names = ["atom", "bond"]
    # set2set_ntypes_direct = None

    model = GatedGCNReactionNetwork(
        in_feats=dataset.feature_size,
        embedding_size=args.embedding_size,
        gated_num_layers=args.gated_num_layers,
        gated_hidden_size=args.gated_hidden_size,
        gated_graph_norm=args.gated_graph_norm,
        gated_batch_norm=args.gated_batch_norm,
        gated_activation=args.gated_activation,
        gated_residual=args.gated_residual,
        gated_dropout=args.gated_dropout,
        num_lstm_iters=args.num_lstm_iters,
        num_lstm_layers=args.num_lstm_layers,
        set2set_ntypes_direct=set2set_ntypes_direct,
        fc_num_layers=args.fc_num_layers,
        fc_hidden_size=args.fc_hidden_size,
        fc_batch_norm=args.fc_batch_norm,
        fc_activation=args.fc_activation,
        fc_dropout=args.fc_dropout,
        outdim=1,
    )
    print(model)

    if args.device is not None:
        model.to(device=args.device)

    if args.post_analysis != "none":
        print(f"\nStart post analysis ({args.post_analysis}) at:", datetime.now())

        # load saved model
        checkpoints_objs = {"model": model}
        load_checkpoints(checkpoints_objs)

        if args.post_analysis == "write_feature":
            # write_feature
            write_features(
                model,
                feature_names,
                {"train": train_loader, "validation": val_loader},
                "feats.tsv",
                "feats_metadata.tsv",
                args.device,
            )
        elif args.post_analysis == "error_analysis":
            loaders = [train_loader, val_loader, test_loader]
            fnames = ["train_error.txt", "val_error.txt", "test_error.txt"]
            for ld, nm in zip(loaders, fnames):
                error_analysis(model, feature_names, ld, nm, args.device)
        else:
            raise ValueError(f"not supported post analysis type: {args.post_analysis}")

        print(f"\nFinish post analysis ({args.post_analysis}) at:", datetime.now())

        # we only do post analysis and do not need to train; so exist here
        sys.exit(0)

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
    sys.stdout.flush()

    t0 = time.time()

    for epoch in range(args.epochs):
        ti = time.time()

        # train
        loss, train_acc = train(
            optimizer, model, feature_names, train_loader, loss_func, metric, args.device
        )

        # bad, we get nan
        if np.isnan(loss):
            print(
                "\n\nBad, we get nan for loss. Turn debug on and hope to catch "
                "it. Note, although we load the checkpoints before the failing, "
                "the debug may still fail because the data loader is random and "
                "thus the data feeded are different from what causes the failing "
                "when running directly. The only hope is that the error can reoccur."
            )
            sys.stdout.flush()
            sys.exit(1)

        # evaluate
        val_acc = evaluate(model, feature_names, val_loader, metric, args.device)

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

    test_acc = evaluate(model, feature_names, test_loader, metric, args.device)

    # write features for post analysis
    write_features(
        model,
        feature_names,
        {"train": train_loader, "validation": val_loader},
        "feats.tsv",
        "feats_metadata.tsv",
        args.device,
    )

    tt = time.time() - t0
    print("\n#TestAcc: {:12.6e} | Total time (s): {:.2f}\n".format(test_acc, tt))
    print("\nFinish training at:", datetime.now())


# do not make it main because we need to run hypertunity
seed_torch()
args = parse_args()
print(args)
main(args)
