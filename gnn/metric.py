import torch
from torch import nn
import warnings


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss that weighs each element differently.

    Weight will be multiplied to the squares of difference.
    All settings are the same as the :meth:`torch.nn.MSELoss`, except that if
    `reduction` is `mean`, the loss will be divided by the sum of `weight`, instead of
    the batch size.

    The weight could be used to ignore some elements in `target` by setting the
    corresponding elements in weight to 0, and it can also be used to scale the element
    differently.
    """

    def __init__(self, reduction="mean"):
        self.reduction = reduction
        super(WeightedMSELoss, self).__init__()

    def forward(self, input, target, weight):

        if input.size() != target.size() != weight.size():
            warnings.warn(
                "Input size ({}) is different from the target size ({}) or weight "
                "size ({}). This will likely lead to incorrect results due "
                "to broadcasting. Please ensure they have the same size.".format(
                    input.size(), target.size(), weight.size()
                )
            )

        rst = ((input - target) ** 2) * weight
        if self.reduction != "none":
            if self.reduction == "mean":
                rst = torch.sum(rst) / torch.sum(weight)
            else:
                rst = torch.sum(rst)

        return rst


class WeightedL1Loss(nn.Module):
    """
    Weighted L1 Loss that weighs each element differently.

    Weight will be multiplied to the squares of difference.
    All settings are the same as the :meth:`torch.nn.L1Loss`, except that if
    `reduction` is `mean`, the loss will be divided by the sum of `weight`, instead of
    the batch size.

    The weight could be used to ignore some elements in `target` by setting the
    corresponding elements in weight to 0, and it can also be used to scale the element
    differently.
    """

    def __init__(self, reduction="mean"):
        self.reduction = reduction
        super(WeightedL1Loss, self).__init__()

    def forward(self, input, target, weight):

        if input.size() != target.size() != weight.size():
            warnings.warn(
                "Input size ({}) is different from the target size ({}) or weight "
                "size ({}). This will likely lead to incorrect results due "
                "to broadcasting. Please ensure they have the same size.".format(
                    input.size(), target.size(), weight.size()
                )
            )

        rst = torch.abs(input - target) * weight
        if self.reduction != "none":
            if self.reduction == "mean":
                rst = torch.sum(rst) / torch.sum(weight)
            else:
                rst = torch.sum(rst)

        return rst


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
