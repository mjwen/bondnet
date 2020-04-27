import sys
import time
import warnings
import torch
import argparse
import numpy as np
from datetime import datetime
from collections import Counter
from torch import autograd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import (
    f1_score,
    classification_report,
    precision_recall_fscore_support,
)
from gnn.metric import EarlyStopping
from gnn.model.hgat_reaction import HGATReaction
from gnn.data.dataset import train_validation_test_split
from gnn.data.electrolyte import ElectrolyteReactionDataset
from gnn.data.dataloader import DataLoaderReaction
from gnn.data.grapher import HeteroMoleculeGraph
from gnn.data.featurizer import (
    AtomFeaturizer,
    BondAsNodeFeaturizer,
    GlobalFeaturizerCharge,
)
from gnn.utils import pickle_dump, seed_torch, load_checkpoints


def parse_args():
    parser = argparse.ArgumentParser(description="HGATReaction")

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

    parser.add_argument(
        "--gat-residual", type=int, default=1, help="residual connection for gat layer"
    )

    parser.add_argument(
        "--gat-batch-norm", type=int, default=0, help="batch norm for gat layer"
    )

    parser.add_argument(
        "--gat-activation", type=str, default="ELU", help="activation fn for gat layer"
    )

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
    parser.add_argument(
        "--fc-batch-norm", type=int, default=0, help="batch nonrm for fc layer"
    )
    parser.add_argument(
        "--fc-activation", type=str, default="ELU", help="activation fn for fc layer"
    )
    parser.add_argument(
        "--fc-drop", type=float, default=0.0, help="dropout rato for fc layer"
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
    all_pred_class = []
    all_target_class = []

    for it, (bg, label) in enumerate(data_loader):
        feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes}
        target_class = label["value"]
        if device is not None:
            feats = {k: v.to(device) for k, v in feats.items()}
            target_class = target_class.to(device)

        pred = model(
            bg,
            feats,
            label["num_mols"],
            label["atom_mapping"],
            label["bond_mapping"],
            label["global_mapping"],
        )
        pred = pred.view(-1)

        # update parameters
        loss = loss_fn(pred, target_class)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()

        # retain data for score computation
        pred_class = [1 if i >= 0.5 else 0 for i in pred]
        all_pred_class.append(pred_class)
        all_target_class.append(target_class.detach().cpu().numpy())

    epoch_loss /= it + 1

    # compute f1 score
    all_pred_class = np.concatenate(all_pred_class)
    all_target_class = np.concatenate(all_target_class)
    if metric_fn == "f1_score":
        score = f1_score(all_target_class, all_pred_class)
    elif metric_fn == "prfs":
        score = precision_recall_fscore_support(all_target_class, all_pred_class)
    elif metric_fn == "classification_report":
        score = classification_report(all_target_class, all_pred_class)
    else:
        raise ValueError("Unsupported metric `{}`".format(metric_fn))

    return epoch_loss, score


def evaluate(model, nodes, data_loader, metric_fn, device=None):
    """
    Evaluate the accuracy of an validation set of test set.

    Args:
        metric_fn (function): the function should be using a `sum` reduction method.
    """
    model.eval()

    with torch.no_grad():

        all_pred_class = []
        all_target_class = []

        for bg, label in data_loader:
            feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes}
            target_class = label["value"]
            if device is not None:
                feats = {k: v.to(device) for k, v in feats.items()}

            pred = model(
                bg,
                feats,
                label["num_mols"],
                label["atom_mapping"],
                label["bond_mapping"],
                label["global_mapping"],
            )
            pred = pred.view(-1)

            # retain data for score computation
            pred_class = [1 if i >= 0.5 else 0 for i in pred]
            all_pred_class.append(pred_class)
            all_target_class.append(target_class.numpy())

    # compute f1 score
    all_pred_class = np.concatenate(all_pred_class)
    all_target_class = np.concatenate(all_target_class)
    if metric_fn == "f1_score":
        score = f1_score(all_target_class, all_pred_class)
    elif metric_fn == "prfs":
        score = precision_recall_fscore_support(all_target_class, all_pred_class)
    elif metric_fn == "classification_report":
        score = classification_report(all_target_class, all_pred_class)
    else:
        raise ValueError("Unsupported metric `{}`".format(metric_fn))

    return score


def score_to_string(score, metric_fn="prfs"):
    if metric_fn == "prfs":
        res = ""
        for i, line in enumerate(score):
            # do not use support
            if i < 3:
                res += " ["
                for j in line:
                    res += "{:.2f} ".format(j)
                res = res[:-1] + "]"
        return res
    else:
        return str(score)


def get_class_weight(data_loader):
    """
    Return a 1D tensor of the weight for positive example (class 1), which is set to
    be equal to the number of negative examples divided by the number os positive
    examples.
    """
    target_class = np.concatenate([label["value"].numpy() for bg, label in data_loader])
    counts = [v for k, v in sorted(Counter(target_class).items())]
    assert len(counts) == 2, f"number of classes {len(counts)} should be 2"

    weight = torch.tensor([counts[0] / counts[1]])

    return weight


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
    print("\n\nStart training at:", datetime.now())

    ### dataset
    sdf_file = "~/Applications/db_access/mol_builder/struct_rxn_clfn_n200.sdf"
    label_file = "~/Applications/db_access/mol_builder/label_rxn_clfn_n200.yaml"
    feature_file = "~/Applications/db_access/mol_builder/feature_rxn_clfn_n200.yaml"

    dataset = ElectrolyteReactionDataset(
        grapher=get_grapher(),
        sdf_file=sdf_file,
        label_file=label_file,
        feature_file=feature_file,
        label_transformer=False,
    )
    trainset, valset, testset = train_validation_test_split(
        dataset, validation=0.1, test=0.1
    )
    print(
        "Trainset size: {}, valset size: {}: testset size: {}.".format(
            len(trainset), len(valset), len(testset)
        )
    )

    train_loader = DataLoaderReaction(trainset, batch_size=args.batch_size, shuffle=True)
    # larger val and test set batch_size is faster but needs more memory
    # adjust the batch size of to fit memory
    bs = max(len(valset) // 10, 1)
    val_loader = DataLoaderReaction(valset, batch_size=bs, shuffle=False)
    bs = max(len(testset) // 10, 1)
    test_loader = DataLoaderReaction(testset, batch_size=bs, shuffle=False)

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
    model = HGATReaction(
        attn_mechanism,
        attn_order,
        in_feats,
        num_gat_layers=args.num_gat_layers,
        gat_hidden_size=args.gat_hidden_size,
        num_heads=args.num_heads,
        feat_drop=args.feat_drop,
        attn_drop=args.attn_drop,
        negative_slope=args.negative_slope,
        gat_residual=args.gat_residual,
        gat_batch_norm=args.gat_batch_norm,
        gat_activation=args.gat_activation,
        num_lstm_iters=args.num_lstm_iters,
        num_lstm_layers=args.num_lstm_layers,
        set2set_ntypes_direct=set2set_ntypes_direct,
        num_fc_layers=args.num_fc_layers,
        fc_hidden_size=args.fc_hidden_size,
        fc_batch_norm=args.fc_batch_norm,
        fc_activation=args.fc_activation,
        fc_drop=args.fc_drop,
        outdim=1,
    )
    print(model)

    if args.device is not None:
        model.to(device=args.device)

    ### optimizer, loss, and metric
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    pos_weight = get_class_weight(train_loader)
    if args.device is not None:
        pos_weight = pos_weight.to(args.device)
    loss_func = BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

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

    print(
        "\n\n# Epoch     Loss         TrainScore(prec,recall,f1)        ValScore("
        "pred,recall,f1)     Time (s)"
    )
    sys.stdout.flush()

    for epoch in range(args.epochs):
        ti = time.time()

        # train and evaluate accuracy
        loss, train_score = train(
            optimizer, model, attn_order, train_loader, loss_func, "prfs", args.device
        )

        # bad, we get nan. Before existing, do some debugging
        if np.isnan(loss):
            print("\n\nBad, we get nan for loss. See below for traceback\n\n")
            sys.stdout.flush()
            with autograd.detect_anomaly():
                train(
                    optimizer,
                    model,
                    attn_order,
                    train_loader,
                    loss_func,
                    "prfs",
                    args.device,
                )
            sys.exit(1)

        val_score = evaluate(model, attn_order, val_loader, "prfs", args.device)

        try:
            recall = val_score[1][1]  # recall of the 1 class
        except IndexError:
            pass

        if stopper.step(-recall, checkpoints_objs, msg="epoch " + str(epoch)):
            # save results for hyperparam tune
            pickle_dump(float(stopper.best_score), args.output_file)
            break

        scheduler.step(-recall)

        tt = time.time() - ti

        print(
            "{:5d}   {:12.6e}   {}   {}   {:.2f}".format(
                epoch, loss, score_to_string(train_score), score_to_string(val_score), tt
            )
        )
        if epoch % 10 == 0:
            sys.stdout.flush()

    # save results for hyperparam tune
    pickle_dump(float(stopper.best_score), args.output_file)

    # load best to calculate test accuracy
    load_checkpoints(checkpoints_objs)
    score = evaluate(model, attn_order, test_loader, "prfs", args.device)
    print("\nTest classification report:")
    print(score)

    print("\nFinish training at:", datetime.now())


# do not make it main because we need to run hypertunity
seed_torch()
args = parse_args()
print(args)
main(args)
