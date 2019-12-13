import torch
from torch import nn
import warnings
from gnn.data.dataloader import DataLoader, DataLoaderQM9


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

    """

    def __init__(self, reduction="mean", scale=1.0):
        self.reduction = reduction
        self.scale = scale
        super(MSELoss, self).__init__()

    def forward(self, input, target):
        """
        Args:
            input (tensor): input of shape (N, *)
            target(dict): with keys `value` and `indicator`, the value associated with
                `value` is a tensor of shape (N, *) and `indocator` is a tensor (0 or
                1) of shape (N,) indicating whether the correspond `input` and `target`
                entries are used to construct the loss.

        Returns:
            Scalar Tensor
        """

        t_value = target["value"]
        t_indicator = target["indicator"]

        if t_value.size() != t_indicator.size() != input.size():
            warnings.warn(
                "Input size ({}) is different to the target value size ({}) or target "
                "indicator size ({}). This will likely lead to incorrect results due "
                "to broadcasting. Please ensure they have the same size.".format(
                    input.size(), t_value.size(), t_indicator.size()
                )
            )

        ret = self.scale * ((input - t_value) ** 2) * t_indicator
        if self.reduction != "none":
            if self.reduction == "mean":
                ret = torch.sum(ret) / torch.sum(t_indicator)
            else:
                ret = torch.sum(ret)

        return ret


class L1Loss(nn.Module):
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
        super(L1Loss, self).__init__()

    def forward(self, input, target):
        """
        Args:
            input (tensor): input of shape (N, *)
            target(dict): with keys `value` and `indicator`, the value associated with
                `value` is a tensor of shape (N, *) and `indocator` is a tensor (0 or
                1) of shape (N,) indicating whether the correspond `input` and `target`
                entries are used to construct the loss.

        Returns:
            Scalar Tensor
        """

        t_value = target["value"]
        t_indicator = target["indicator"]

        if t_value.size() != t_indicator.size() != input.size():
            warnings.warn(
                "Input size ({}) is different to the target value size ({}) or target "
                "indicator size ({}). This will likely lead to incorrect results due "
                "to broadcasting. Please ensure they have the same size.".format(
                    input.size(), t_value.size(), t_indicator.size()
                )
            )

        ret = torch.abs(input - t_value) * t_indicator
        if self.reduction != "none":
            if self.reduction == "mean":
                ret = torch.sum(ret) / torch.sum(t_indicator)
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
            self.save_checkpoint(model, msg, score)
        elif score < self.best_score:
            self.best_score = score
            self.save_checkpoint(model, msg, score)
            self.counter = 0
        else:
            self.counter += 1
            if not self.silent:
                print("EarlyStopping counter: {}/{}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    @staticmethod
    def save_checkpoint(model, msg, score):
        """
        Saves model when validation loss decrease.
        """
        torch.save(model.state_dict(), "es_checkpoint.pkl")
        with open("es_message.log", "w") as f:
            f.write("{}  {}".format(msg, score))


def evaluate(model, data_loader, metric_fn, nodes, device=None):
    """
    Evaluate the accuracy of a dataset for a given metric specified by the metric_fn.
    Note, the metric should measure the ``mean`` accuracy of a batch of data. This
    function will measure the mean accuracy across all data points.

    Args:
        model (callable): the model to compute prediction
        data_loader (torch.utils.data.DataLoader): loader for the data
        metric_fn (callable): a metric function to evaluate the accuracy
        nodes (list of str): the graph nodes on which feats reside
        device (str): to device to perform the computation. e.g. `cpu`, `cuda`

    Returns:
        float: accuracy
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
            if isinstance(label, dict):
                c = sum(label["indicator"])
            else:
                c = len(label)
            accuracy += metric_fn(pred, label) * c
            count += c
    return accuracy / count
