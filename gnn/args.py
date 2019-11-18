import argparse


def create_parser():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument(
        "--gpu", type=int, default=-1, help="which GPU to use. Set -1 to use CPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="number of training epochs"
    )
    parser.add_argument(
        "--num-heads", type=int, default=8, help="number of hidden attention heads"
    )
    parser.add_argument(
        "--num-out-heads", type=int, default=1, help="number of output attention heads"
    )
    parser.add_argument(
        "--num-layers", type=int, default=1, help="number of hidden layers"
    )
    parser.add_argument(
        "--num-hidden", type=int, default=8, help="number of hidden units"
    )
    parser.add_argument(
        "--residual", action="store_true", default=False, help="use residual connection"
    )
    parser.add_argument(
        "--in-drop", type=float, default=0.6, help="input feature dropout"
    )
    parser.add_argument("--attn-drop", type=float, default=0.6, help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument(
        "--negative-slope",
        type=float,
        default=0.2,
        help="the negative slope of leaky relu",
    )
    parser.add_argument(
        "--fastmode",
        action="store_true",
        default=False,
        help="skip re-evaluate the validation set",
    )
    args = parser.parse_args()
    # for k, v in vars(args).items():
    #     print(k, v)
    return args
