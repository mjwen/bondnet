import torch
from torch import nn
import warnings
from gnn.data.dataloader import DataLoader


class MSELoss(nn.Module):
    r"""
    Mean squared loss between input and target with an additional binary argument
    `indicator` to ignore some contributions.

    The unreduced (with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2 \dot z_n,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{sum}(L)/\operatorname{sum}(indicators), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each, and :math:`z` is a 1D tensor that should be of the same
    length as :math:`y`.

    Args:
        reduction (str): reduction mechanism
        scale (float): to scale the loss by this constant, merely for numerical stability

    Shapes:
        input: (N, *)
        target: (N, *)
        indicator: (N,)

    Returns:
        0D tensor of the results
    """

    def __init__(self, reduction="mean", scale=1.0):
        self.reduction = reduction
        self.scale = scale
        super(MSELoss, self).__init__()

    def forward(self, input, target, indicator):

        if target.size() != input.size():
            warnings.warn(
                "Using a target size ({}) that is different to the input size ({}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.".format(
                    target.size(), input.size()
                )
            )

        if len(target) != len(indicator):
            raise ValueError(
                "Using indicator length({}) that is different from target length({}). "
                "They should of the same length".format(len(indicator), len(target))
            )

        ret = self.scale * ((input - target) ** 2) * indicator
        if self.reduction != "none":
            if self.reduction == "mean":
                ret = torch.sum(ret) / torch.sum(indicator)
            else:
                ret = torch.sum(ret)

        return ret


class MAELoss(nn.Module):
    r"""
    Mean absolute error between input and target with an additional binary argument
    `indicator` to ignore some contributions.

    The unreduced (with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \| x_n - y_n \| \dot z_n,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{sum}(L)/\operatorname{sum}(indicators), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each, and :math:`z` is a 1D tensor that should be of the same
    length as :math:`y`.

    Args:
        reduction (str): reduction mechanism

    Shapes:
        input: (N, *)
        target: (N, *)
        indicator: (N,)

    Returns:
        0D tensor of the results
    """

    def __init__(self, reduction="mean"):
        self.reduction = reduction
        super(MAELoss, self).__init__()

    def forward(self, input, target, indicator):

        if target.size() != input.size():
            warnings.warn(
                "Using a target size ({}) that is different to the input size ({}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.".format(
                    target.size(), input.size()
                )
            )

        if len(target) != len(indicator):
            raise ValueError(
                "Using indicator length({}) that is different from target length({}). "
                "They should of the same length".format(len(indicator), len(target))
            )

        ret = torch.abs(input - target) * indicator
        if self.reduction != "none":
            if self.reduction == "mean":
                ret = torch.sum(ret) / torch.sum(indicator)
            else:
                ret = torch.sum(ret)

        return ret


class EarlyStopping:
    def __init__(self, patience=200, silent=True):
        self.patience = patience
        self.silent = silent
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model, msg=None):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, msg)
        elif score > self.best_score:
            self.counter += 1
            if not self.silent:
                print("EarlyStopping counter: {}/{}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, msg)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model, msg):
        """
        Saves model when validation loss decrease.
        """
        torch.save(model.state_dict(), "es_checkpoint.pkl")
        with open("es_message.log", "w") as f:
            f.write(str(msg))


def evaluate(model, dataset, metric_fn, nodes, device=None):
    """
    Evaluate the accuracy of a dataset for a given metric specified by the metric_fn.

    Args:
        model (callable): the model to compute prediction
        dataset: the dataset
        metric_fn (callable): a metric function to evaluate the accuracy
        nodes (list of str): the graph nodes on which feats reside
        device (str): to device to perform the computation. e.g. `cpu`, `cuda`

    Returns:
        float: accuracy
    """
    model.eval()
    size = len(dataset)  # whole dataset
    data_loader = DataLoader(dataset, batch_size=size, shuffle=False)

    with torch.no_grad():
        for bg, label in data_loader:
            feats = {nt: bg.nodes[nt].data["feat"] for nt in nodes}
            if device is not None:
                feats = {k: v.to(device) for k, v in feats.items()}
                label = {k: v.to(device) for k, v in label.items()}
            pred = model(bg, feats)
            accuracy = metric_fn(pred, label["energies"], label["indicators"])
            return accuracy
