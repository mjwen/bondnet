# pylint: disable=no-member
import sys
import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from gnn.metric import WeightedMSELoss, WeightedL1Loss
from gnn.data.dataset import train_validation_test_split
from gnn.data.electrolyte import ElectrolyteDataset
from gnn.data.dataloader import DataLoaderElectrolyte
from gnn.model.hgat import HGAT
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
    count = 0.0

    for it, (bg, label, scale) in enumerate(data_loader):
        feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes}
        if device is not None:
            feats = {k: v.to(device=device) for k, v in feats.items()}
            label = {k: v.to(device=device) for k, v in label.items()}
        label_val = label["value"]
        label_ind = label["indicator"]

        pred = model(bg, feats)
        loss = loss_fn(pred, label_val, label_ind)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()
        weight = label_ind * scale
        accuracy += metric_fn(pred, label_val, weight).detach().item()
        count += sum(label_ind).item()

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
                label = {k: v.to(device) for k, v in label.items()}
            label_val = label["value"]
            label_ind = label["indicator"]

            pred = model(bg, feats)

            weight = label_ind * scale
            accuracy += metric_fn(pred, label_val, weight).detach().item()
            count += sum(label_ind).item()

    return accuracy / count


def main(args):
    # dataset
    sdf_file = "/Users/mjwen/Applications/mongo_db_access/extracted_data/sturct_n200.sdf"
    label_file = "/Users/mjwen/Applications/mongo_db_access/extracted_data/label_n200.txt"
    dataset = ElectrolyteDataset(sdf_file, label_file)
    trainset, valset, testset = train_validation_test_split(
        dataset, validation=0.1, test=0.1
    )
    train_loader = DataLoaderElectrolyte(
        trainset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoaderElectrolyte(valset, batch_size=len(valset), shuffle=False)
    test_loader = DataLoaderElectrolyte(testset, batch_size=len(testset), shuffle=False)

    # model
    attn_mechanism = {
        "atom": {"edges": ["b2a", "g2a", "a2a"], "nodes": ["bond", "global", "atom"]},
        "bond": {"edges": ["a2b", "g2b", "b2b"], "nodes": ["atom", "global", "bond"]},
        "global": {"edges": ["a2g", "b2g", "g2g"], "nodes": ["atom", "bond", "global"]},
    }
    attn_order = ["atom", "bond", "global"]
    in_feats = trainset.get_feature_size(attn_order)
    model = HGAT(
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

    # optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_func = WeightedMSELoss(reduction="mean")
    metric = WeightedL1Loss(reduction="sum")

    # learning rate scheduler and stopper
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
