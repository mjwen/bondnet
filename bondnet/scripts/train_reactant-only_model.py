import sys
import time
import warnings
import torch
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from bondnet.scripts.metric import WeightedL1Loss, EarlyStopping
from bondnet.model.gated_bond import GatedGCNBond
from bondnet.data.dataset import train_validation_test_split
from bondnet.data.electrolyte import ElectrolyteBondDataset
from bondnet.data.dataloader import DataLoaderBond
from bondnet.data.grapher import HeteroMoleculeGraph
from bondnet.data.featurizer import (
    AtomFeaturizerFull,
    BondAsNodeFeaturizerFull,
    GlobalFeaturizer,
)
from bondnet.utils import (
    load_checkpoints,
    save_checkpoints,
    seed_torch,
    pickle_dump,
    yaml_dump,
)

best = np.finfo(np.float32).max


def parse_args():
    parser = argparse.ArgumentParser(description="GatedBond")

    # embedding layer
    parser.add_argument("--embedding-size", type=int, default=24)

    # gated layer
    parser.add_argument("--gated-num-layers", type=int, default=3)
    parser.add_argument("--gated-hidden-size", type=int, nargs="+", default=[64, 64, 64])
    parser.add_argument("--gated-num-fc-layers", type=int, default=1)
    parser.add_argument("--gated-graph-norm", type=int, default=0)
    parser.add_argument("--gated-batch-norm", type=int, default=1)
    parser.add_argument("--gated-activation", type=str, default="ReLU")
    parser.add_argument("--gated-residual", type=int, default=1)
    parser.add_argument("--gated-dropout", type=float, default="0.0")

    # readout layer
    parser.add_argument("--readout-type", type=str, default="bond")

    # fc layer
    parser.add_argument("--fc-num-layers", type=int, default=2)
    parser.add_argument("--fc-hidden-size", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--fc-batch-norm", type=int, default=0)
    parser.add_argument("--fc-activation", type=str, default="ReLU")
    parser.add_argument("--fc-dropout", type=float, default=0.0)

    # training
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=100, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--restore", type=int, default=0, help="read checkpoints")
    parser.add_argument(
        "--dataset-state-dict-filename", type=str, default="dataset_state_dict.pkl"
    )
    # gpu
    parser.add_argument(
        "--gpu", type=int, default=None, help="GPU index. None to use CPU."
    )
    parser.add_argument(
        "--distributed",
        type=int,
        default=0,
        help="DDP training, --gpu is ignored if this is True",
    )
    parser.add_argument(
        "--num-gpu",
        type=int,
        default=None,
        help="Number of GPU to use in distributed mode; ignored otherwise.",
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://localhost:13456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--dist-backend", type=str, default="nccl")

    # output file (needed by hypertunity)
    parser.add_argument("--output_file", type=str, default="results.pkl")

    args = parser.parse_args()

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
        args.fc_hidden_size = [max(val // 2 ** i, 8) for i in range(args.fc_num_layers)]
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
        index = label["index"]
        # norm_atom = label["norm_atom"]
        # norm_bond = label["norm_bond"]
        norm_atom = None
        norm_bond = None
        try:
            stdev = label["scaler_stdev"]
        except KeyError:
            stdev = None

        if device is not None:
            feats = {k: v.to(device) for k, v in feats.items()}
            target = target.to(device)
            index = index.to(device)
            # norm_atom = norm_atom.to(device)
            # norm_bond = norm_bond.to(device)
            if stdev is not None:
                stdev = stdev.to(device)

        pred = model(bg, feats, norm_atom, norm_bond)
        pred = pred[index]

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

        for it, (bg, label) in enumerate(data_loader):
            feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes}
            target = label["value"]
            index = label["index"]
            # norm_atom = label["norm_atom"]
            # norm_bond = label["norm_bond"]
            norm_atom = None
            norm_bond = None
            try:
                stdev = label["scaler_stdev"]
            except KeyError:
                stdev = None

            if device is not None:
                feats = {k: v.to(device) for k, v in feats.items()}
                target = target.to(device)
                index = index.to(device)
                # norm_atom = norm_atom.to(device)
                # norm_bond = norm_bond.to(device)
                if stdev is not None:
                    stdev = stdev.to(device)

            pred = model(bg, feats, norm_atom, norm_bond)
            pred = pred[index]

            accuracy += metric_fn(pred, target, stdev).detach().item()
            count += len(target)

    return accuracy / count


def get_grapher():
    atom_featurizer = AtomFeaturizerFull()
    bond_featurizer = BondAsNodeFeaturizerFull(length_featurizer=None, dative=False)
    global_featurizer = GlobalFeaturizer(allowed_charges=None)

    # atom_featurizer = AtomFeaturizerMinimum()
    # bond_featurizer = BondAsNodeFeaturizerMinimum(length_featurizer=None)
    # global_featurizer = GlobalFeaturizer(allowed_charges=[-1, 0, 1])

    grapher = HeteroMoleculeGraph(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        global_featurizer=global_featurizer,
        self_loop=True,
    )
    return grapher


def main_worker(gpu, world_size, args):
    global best
    args.gpu = gpu

    if not args.distributed or (args.distributed and args.gpu == 0):
        print("\n\nStart training at:", datetime.now())

    if args.distributed:
        dist.init_process_group(
            args.dist_backend,
            init_method=args.dist_url,
            world_size=world_size,
            rank=args.gpu,
        )

    # Explicitly setting seed to ensure the same dataset split and models created in
    # two processes (when distributed) start from the same random weights and biases
    seed_torch()

    ### dataset
    sdf_file = "~/Applications/db_access/mol_builder/struct_n200.sdf"
    label_file = "~/Applications/db_access/mol_builder/label_n200.yaml"
    feature_file = "~/Applications/db_access/mol_builder/feature_n200.yaml"
    # sdf_file = "~/Applications/db_access/zinc_bde/zinc_struct_bond_rgrn_n200.sdf"
    # label_file = "~/Applications/db_access/zinc_bde/zinc_label_bond_rgrn_n200.txt"
    # feature_file = "~/Applications/db_access/zinc_bde/zinc_feature_bond_rgrn_n200.yaml"

    if args.restore:
        dataset_state_dict_filename = args.dataset_state_dict_filename

        if dataset_state_dict_filename is None:
            warnings.warn("Restore with `args.dataset_state_dict_filename` set to None.")
        elif not Path(dataset_state_dict_filename).exists():
            warnings.warn(
                f"`{dataset_state_dict_filename} not found; set "
                f"args.dataset_state_dict_filename` to None"
            )
            dataset_state_dict_filename = None
    else:
        dataset_state_dict_filename = None

    dataset = ElectrolyteBondDataset(
        grapher=get_grapher(),
        molecules=sdf_file,
        labels=label_file,
        extra_features=feature_file,
        feature_transformer=True,
        label_transformer=True,
        state_dict_filename=dataset_state_dict_filename,
    )

    trainset, valset, testset = train_validation_test_split(
        dataset, validation=0.1, test=0.1
    )

    if not args.distributed or (args.distributed and args.gpu == 0):
        torch.save(dataset.state_dict(), args.dataset_state_dict_filename)
        print(
            "Trainset size: {}, valset size: {}: testset size: {}.".format(
                len(trainset), len(valset), len(testset)
            )
        )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        train_sampler = None

    train_loader = DataLoaderBond(
        trainset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )
    # larger val and test set batch_size is faster but needs more memory
    # adjust the batch size of to fit memory
    bs = max(len(valset) // 10, 1)
    val_loader = DataLoaderBond(valset, batch_size=bs, shuffle=False)
    bs = max(len(testset) // 10, 1)
    test_loader = DataLoaderBond(testset, batch_size=bs, shuffle=False)

    ### model

    feature_names = ["atom", "bond", "global"]
    set2set_ntypes_direct = ["global"]
    feature_size = dataset.feature_size

    # feature_names = ["atom", "bond"]
    # set2set_ntypes_direct = None
    # feature_size = {k: v for k, v in dataset.feature_size.items() if k != "global"}

    args.feature_size = feature_size
    args.set2set_ntypes_direct = set2set_ntypes_direct

    # save args
    if not args.distributed or (args.distributed and args.gpu == 0):
        yaml_dump(args, "train_args.yaml")

    model = GatedGCNBond(
        in_feats=args.feature_size,
        embedding_size=args.embedding_size,
        gated_num_layers=args.gated_num_layers,
        gated_hidden_size=args.gated_hidden_size,
        gated_num_fc_layers=args.gated_num_fc_layers,
        gated_graph_norm=args.gated_graph_norm,
        gated_batch_norm=args.gated_batch_norm,
        gated_activation=args.gated_activation,
        gated_residual=args.gated_residual,
        gated_dropout=args.gated_dropout,
        readout_type=args.readout_type,
        fc_num_layers=args.fc_num_layers,
        fc_hidden_size=args.fc_hidden_size,
        fc_batch_norm=args.fc_batch_norm,
        fc_activation=args.fc_activation,
        fc_dropout=args.fc_dropout,
        outdim=1,
    )

    if not args.distributed or (args.distributed and args.gpu == 0):
        print(model)

    if args.gpu is not None:
        model.to(args.gpu)
    if args.distributed:
        ddp_model = DDP(model, device_ids=[args.gpu], find_unused_parameters=False)
        ddp_model.feature_before_fc = model.feature_before_fc
        model = ddp_model

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

    # load checkpoint
    state_dict_objs = {"model": model, "optimizer": optimizer, "scheduler": scheduler}
    if args.restore:
        try:

            if args.gpu is None:
                checkpoint = load_checkpoints(state_dict_objs, filename="checkpoint.pkl")
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = load_checkpoints(
                    state_dict_objs, map_location=loc, filename="checkpoint.pkl"
                )

            args.start_epoch = checkpoint["epoch"]
            best = checkpoint["best"]
            print(f"Successfully load checkpoints, best {best}, epoch {args.start_epoch}")

        except FileNotFoundError as e:
            warnings.warn(str(e) + " Continue without loading checkpoints.")
            pass

    # start training
    if not args.distributed or (args.distributed and args.gpu == 0):
        print("\n\n# Epoch     Loss         TrainAcc        ValAcc     Time (s)")
        sys.stdout.flush()

    for epoch in range(args.start_epoch, args.epochs):
        ti = time.time()

        # In distributed mode, calling the set_epoch method is needed to make shuffling
        # work; each process will use the same random seed otherwise.
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train
        loss, train_acc = train(
            optimizer, model, feature_names, train_loader, loss_func, metric, args.gpu
        )

        # bad, we get nan
        if np.isnan(loss):
            print("\n\nBad, we get nan for loss. Existing")
            sys.stdout.flush()
            sys.exit(1)

        # evaluate
        val_acc = evaluate(model, feature_names, val_loader, metric, args.gpu)

        if stopper.step(val_acc):
            pickle_dump(best, args.output_file)  # save results for hyperparam tune
            break

        scheduler.step(val_acc)

        is_best = val_acc < best
        if is_best:
            best = val_acc

        # save checkpoint
        if not args.distributed or (args.distributed and args.gpu == 0):

            misc_objs = {"best": best, "epoch": epoch}

            save_checkpoints(
                state_dict_objs,
                misc_objs,
                is_best,
                msg=f"epoch: {epoch}, score {val_acc}",
            )

            tt = time.time() - ti

            print(
                "{:5d}   {:12.6e}   {:12.6e}   {:12.6e}   {:.2f}".format(
                    epoch, loss, train_acc, val_acc, tt
                )
            )
            if epoch % 10 == 0:
                sys.stdout.flush()

    # load best to calculate test accuracy
    if args.gpu is None:
        checkpoint = load_checkpoints(state_dict_objs, filename="best_checkpoint.pkl")
    else:
        # Map model to be loaded to specified single  gpu.
        loc = "cuda:{}".format(args.gpu)
        checkpoint = load_checkpoints(
            state_dict_objs, map_location=loc, filename="best_checkpoint.pkl"
        )

    if not args.distributed or (args.distributed and args.gpu == 0):
        test_acc = evaluate(model, feature_names, test_loader, metric, args.gpu)

        print("\n#TestAcc: {:12.6e} \n".format(test_acc))
        print("\nFinish training at:", datetime.now())


def main():
    args = parse_args()
    print(args)

    if args.distributed:
        # DDP
        world_size = torch.cuda.device_count() if args.num_gpu is None else args.num_gpu
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))

    else:
        # train on CPU or a single GPU
        main_worker(args.gpu, None, args)


if __name__ == "__main__":
    main()
