import argparse


def create_parser():
    parser = argparse.ArgumentParser(description="HGAT")

    # model
    parser.add_argument(
        "--num-gat-layers", type=int, default=3, help="number of GAT layers"
    )
    parser.add_argument(
        "--gat-hidden-size",
        type=int,
        nargs="+",
        default=[32, 32, 32],
        help="number of hidden units of GAT layers",
    )
    parser.add_argument(
        "--num-heads", type=int, default=8, help="number of hidden attention heads"
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
        default=[32, 32, 32],
        help="number of hidden units of fc layers",
    )

    # training
    parser.add_argument(
        "--gpu", type=int, default=-1, help="which GPU to use. Set -1 to use CPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")

    # output file (needed by hypertunity)
    parser.add_argument(
        "--output_file", type=str, default="results.pkl", help="name of output file"
    )

    args = parser.parse_args()

    if len(args.gat_hidden_size) == 1:
        args.gat_hidden_size = args.gat_hidden_size * args.num_gat_layers
    else:
        assert len(args.gat_hidden_size) == args.num_gat_layers, (
            "length of `gat-hidden-size` should be equal to `num-gat-layers`, but got "
            "{} and {}.".format(args.gat_hidden_size, args.num_gat_layers)
        )

    if len(args.fc_hidden_size) == 1:
        args.fc_hidden_size = args.fc_hidden_size * args.num_fc_layers
    assert len(args.fc_hidden_size) == args.num_fc_layers, (
        "length of `fc-hidden-size` should be equal to `num-fc-layers`, but got "
        "{} and {}.".format(args.fc_hidden_size, args.num_fc_layers)
    )
    # for k, v in vars(args).items():
    #     print(k, v)
    return args
