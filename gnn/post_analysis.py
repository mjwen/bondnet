import numpy as np
from gnn.utils import expand_path
import matplotlib.pyplot as plt


def write_error(predictions, targets, ids, sort=True, filename="error.txt"):
    """
    Write the error to file.

    Args:
        predictions (list): model prediction.
        targets (list): reference value.
        ids (list): ids associated with errors.
        sort (bool): whether to sort the error from low to high.
        filename (str): filename to write out the result.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    errors = predictions - targets

    if sort:
        errors, predictions, targets, ids = zip(
            *sorted(zip(errors, predictions, targets, ids), key=lambda pair: pair[0])
        )
    with open(expand_path(filename), "w") as f:
        f.write("# error    prediction    target    id\n")
        for e, p, t, i in zip(errors, predictions, targets, ids):
            f.write("{:13.5e} {:13.5e} {:13.5e}    {}\n".format(e, p, t, i))

        # MAE, MAX Error and RMSE
        abs_e = np.abs(errors)
        mae = np.mean(abs_e)
        rmse = np.sqrt(np.mean(np.square(errors)))
        max_e_idx = np.argmax(abs_e)

        f.write("\n")
        f.write(f"# MAE: {mae}\n")
        f.write(f"# RMSE: {rmse}\n")
        f.write(f"# MAX error: {abs_e[max_e_idx]}   {ids[max_e_idx]}\n")


def read_data(filename, id_col=-1):
    """Read data as dict. Keys should be specified in the first line.

    Returns:
         dict: keys specified in the first line and each column is the values.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # header
    keys = lines[0].strip("#\n").split()

    # body
    data = []
    for line in lines[1:]:
        # strip head and rear whitespaces (including ' \t\n\r\x0b\x0c')
        line = line.strip()
        # delete empty line and comments line beginning with `#'
        if not line or line[0] == "#":
            continue
        line = line.split()
        data.append([item for item in line])
    data = np.asarray(data)

    # convert to dict
    if id_col == -1:
        id_col = len(keys) - 1

    data_dict = dict()
    for i, k in enumerate(keys):
        if i == id_col:
            data_dict[k] = np.array(data[:, i], dtype="object")
        else:
            data_dict[k] = np.array(data[:, i], dtype=np.float64)

    return data_dict


def plot_prediction_vs_target(filename, plot_name="pred_vs_target.pdf"):
    """
    Plot prediction vs target as dots, to show how far they are away from y = x.

    Args:
        filename (str): file contains the data
        plot_name (str): name of the plot file
    """

    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.gca(aspect="auto")

    data = read_data(filename)
    X = data["target"]
    Y = data["prediction"]

    xy_min = min(min(X), min(Y)) - 5
    xy_max = max(max(X), max(Y)) + 5
    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)

    # plot dots
    ax.scatter(X, Y, marker="o", ec=None, alpha=0.6)

    # plot y = x
    ax.plot([xy_min, xy_max], [xy_min, xy_max], "--", color="gray", alpha=0.8)

    # label
    ax.set_xlabel("target")
    ax.set_ylabel("prediction")

    plot_name = expand_path(plot_name)
    fig.savefig(plot_name, bbox_inches="tight")


if __name__ == "__main__":
    plot_prediction_vs_target("train_error.txt", plot_name="pred_vs_target_train.pdf")
    plot_prediction_vs_target("val_error.txt", plot_name="pred_vs_target_val.pdf")
    plot_prediction_vs_target("test_error.txt", plot_name="pred_vs_target_test.pdf")
