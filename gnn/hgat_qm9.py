import sys
import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss, L1Loss
from gnn.model.hgatmol import HGATMol
from gnn.data.dataset import train_validation_test_split
from gnn.data.qm9 import QM9Dataset
from gnn.data.dataloader import DataLoaderQM9
from gnn.metric import EarlyStopping
from gnn.args import parse_args
from gnn.utils import pickle_dump, seed_torch


def train(optimizer, model, nodes, data_loader, loss_fn, metric_fn, device=None):
    """
    Args:
        metric_fn (function): the function should be using a `sum` reduction method.
    """

    model.train()

    epoch_loss = 0.0
    accuracy = 0.0
    count = 0

    for it, (bg, label) in enumerate(data_loader):
        feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes}
        if device is not None:
            feats = {k: v.to(device=device) for k, v in feats.items()}
            if isinstance(label, dict):
                label = {k: v.to(device) for k, v in label.items()}
            else:
                label = label.to(device=device)

        pred = model(bg, feats)
        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()
        accuracy += metric_fn(pred, label).detach().item()
        if isinstance(label, dict):
            count += sum(label["indicator"])
        else:
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
        count = 0

        for bg, label in data_loader:
            feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes}
            if device is not None:
                feats = {k: v.to(device) for k, v in feats.items()}
                if isinstance(label, dict):
                    label = {k: v.to(device) for k, v in label.items()}
                else:
                    label = label.to(device)

            pred = model(bg, feats)
            accuracy += metric_fn(pred, label).detach().item()
            if isinstance(label, dict):
                count += sum(label["indicator"])
            else:
                count += len(label)

    return accuracy / count


def main(args):

    # dataset
    sdf_file = "/Users/mjwen/Documents/Dataset/qm9/gdb9_n200.sdf"
    label_file = "/Users/mjwen/Documents/Dataset/qm9/gdb9_n200.sdf.csv"
    props = ["u0_atom"]
    dataset = QM9Dataset(
        sdf_file,
        label_file,
        self_loop=True,
        grapher="hetero",
        properties=props,
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

    train_loader = DataLoaderQM9(
        trainset, hetero=True, batch_size=args.batch_size, shuffle=True
    )
    # larger val and test set batch_size is faster but needs more memory
    # adjust the batch size of to fit memory
    bs = len(valset) // 4
    val_loader = DataLoaderQM9(valset, hetero=True, batch_size=bs, shuffle=False)
    bs = len(testset) // 4
    test_loader = DataLoaderQM9(testset, hetero=True, batch_size=bs, shuffle=False)

    # model
    attn_mechanism = {
        "atom": {"edges": ["b2a", "g2a", "a2a"], "nodes": ["bond", "global", "atom"]},
        "bond": {"edges": ["a2b", "g2b", "b2b"], "nodes": ["atom", "global", "bond"]},
        "global": {"edges": ["a2g", "b2g", "g2g"], "nodes": ["atom", "bond", "global"]},
    }
    attn_order = ["atom", "bond", "global"]
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
    )
    print(model)
    if args.device is not None:
        model.to(device=args.device)

    # optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # loss, accuracy metric
    loss_func = MSELoss(reduction="mean")
    metric = L1Loss(reduction="sum")

    # learning rate scheduler, and stopper
    patience = 150
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=patience // 3, verbose=True
    )
    stopper = EarlyStopping(patience=patience)

    print("\n\n# Epoch     Loss         TrainAcc        ValAcc     Time (s)")
    t0 = time.time()

    for epoch in range(args.epochs):
        ti = time.time()

        # train and evaluate accuracy
        loss, train_acc = train(
            optimizer, model, attn_order, train_loader, loss_func, metric, args.device
        )
        val_acc = evaluate(model, attn_order, val_loader, metric, args.device)

        scheduler.step(val_acc)
        if stopper.step(val_acc, model, msg="epoch " + str(epoch)):
            # save results for hyperparam tune
            pickle_dump(float(stopper.best_score), args.output_file)
            break
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
    model.load_state_dict(torch.load("es_checkpoint.pkl"))
    test_acc = evaluate(model, attn_order, test_loader, metric, args.device)
    tt = time.time() - t0
    print("\n#TestAcc: {:12.6e} | Total time (s): {:.2f}\n".format(test_acc, tt))


# do not make it make because we need to run hypertunity
seed_torch()
args = parse_args()
print(args)
main(args)
