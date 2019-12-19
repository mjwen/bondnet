import sys
import time
import warnings
import argparse
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss
from gnn.metric import WeightedL1Loss, EarlyStopping
from dgl.model_zoo.chem.mpnn import MPNNModel
from gnn.data.dataset import train_validation_test_split
from gnn.data.qm9 import QM9Dataset
from gnn.data.dataloader import DataLoaderQM9
from gnn.utils import pickle_dump, seed_torch, load_checkpoints


def parse_args():
    parser = argparse.ArgumentParser(description="MPNN")

    # model
    parser.add_argument("--node-hidden-dim", type=int, default=64, help="")
    parser.add_argument("--edge-hidden-dim", type=int, default=128, help="")
    parser.add_argument("--num-step-message-passing", type=int, default=6, help="")
    parser.add_argument("--num-step-set2set", type=int, default=6, help="")
    parser.add_argument("--num-layer-set2set", type=int, default=3, help="")

    # training
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index. -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=100, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--restore", type=int, default=1, help="read checkpoints")

    # output file (needed by hypertunity)
    parser.add_argument(
        "--output_file", type=str, default="results.pkl", help="name of output file"
    )

    args = parser.parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        args.device = torch.device("cuda:{}".format(args.gpu))
    else:
        args.device = None

    return args


def train(optimizer, model, data_loader, loss_fn, metric_fn, device=None):
    """
    Args:
        metric_fn (function): the function should be using a `sum` reduction method.
    """

    model.train()

    epoch_loss = 0.0
    accuracy = 0.0
    count = 0.0

    for it, (bg, label, scale) in enumerate(data_loader):
        nf = bg.ndata["feat"]
        ef = bg.edata["feat"]
        if device is not None:
            nf = nf.to(device=device)
            ef = ef.to(device=device)
            label = label.to(device=device)

        pred = model(bg, nf, ef)
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


def evaluate(model, data_loader, metric_fn, device=None):
    """
    Evaluate the accuracy of an validation set of test set.

    Args:
        metric_fn (function): the function should be using a `sum` reduction method.
    """
    model.eval()

    with torch.no_grad():
        accuracy = 0.0
        count = 0

        for bg, label, scale in data_loader:
            nf = bg.ndata["feat"]
            ef = bg.edata["feat"]
            if device is not None:
                nf = nf.to(device=device)
                ef = ef.to(device=device)
                label = label.to(device=device)

            pred = model(bg, nf, ef)
            accuracy += metric_fn(pred, label, scale).detach().item()
            count += len(label)

    return accuracy / count


def main(args):

    ### dataset
    sdf_file = "/Users/mjwen/Documents/Dataset/qm9/gdb9_n200.sdf"
    label_file = "/Users/mjwen/Documents/Dataset/qm9/gdb9_n200.sdf.csv"
    props = ["u0_atom"]
    dataset = QM9Dataset(
        sdf_file,
        label_file,
        self_loop=False,
        grapher="homo_complete",
        bond_length_featurizer="bin",
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
        trainset, hetero=False, batch_size=args.batch_size, shuffle=True
    )
    # larger val and test set batch_size is faster but needs more memory
    # adjust the batch size of to fit memory
    bs = len(valset) // 10
    val_loader = DataLoaderQM9(valset, hetero=False, batch_size=bs, shuffle=False)
    bs = len(testset) // 10
    test_loader = DataLoaderQM9(testset, hetero=False, batch_size=bs, shuffle=False)

    ### model
    in_feats = trainset.get_feature_size(["atom", "bond"])
    model = MPNNModel(
        node_input_dim=in_feats[0],
        edge_input_dim=in_feats[1],
        output_dim=len(props),
        node_hidden_dim=args.node_hidden_dim,
        edge_hidden_dim=args.edge_hidden_dim,
        num_step_message_passing=args.num_step_message_passing,
        num_step_set2set=args.num_step_set2set,
        num_layer_set2set=args.num_layer_set2set,
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

    # for epoch in range(args.epochs):
    for epoch in range(2):
        ti = time.time()

        # train and evaluate accuracy
        loss, train_acc = train(
            optimizer, model, train_loader, loss_func, metric, args.device
        )
        val_acc = evaluate(model, val_loader, metric, args.device)

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

    # load best to calculate test accuracy
    load_checkpoints(checkpoints_objs)
    test_acc = evaluate(model, test_loader, metric, args.device)

    tt = time.time() - t0
    print("\n#TestAcc: {:12.6e} | Total time (s): {:.2f}\n".format(test_acc, tt))


# do not make it main because we need to run hypertunity
seed_torch()
args = parse_args()
print(args)
main(args)
