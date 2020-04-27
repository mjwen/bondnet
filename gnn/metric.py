import warnings
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


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

    Args:
        weight (Tensor): is weight is `None` this behaves exactly the same as
        :meth:`torch.nn.MSELoss`.
    """

    def __init__(self, reduction="mean"):
        self.reduction = reduction
        super(WeightedMSELoss, self).__init__()

    def forward(self, input, target, weight):

        if weight is None:
            return F.l1_loss(input, target, reduction=self.reduction)
        else:
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

    Args:
        weight (Tensor): is weight is `None` this behaves exactly the same as
        :meth:`torch.nn.L1Loss`.
    """

    def __init__(self, reduction="mean"):
        self.reduction = reduction
        super(WeightedL1Loss, self).__init__()

    def forward(self, input, target, weight):

        if weight is None:
            return F.l1_loss(input, target, reduction=self.reduction)
        else:
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


class OrderAccuracy:
    """
    Order energies of bonds from the same molecule and compute the first `max_n`
    hit accuracy.
    """

    def __init__(self, max_n=3):
        self.max_n = max_n

    def step(self, predictions, targets, mol_sources):
        """

        Args:
            predictions (list): prediction made by the model
            targets (list): target of the prediction
            mol_sources (list): identifier (str or int) indicating the source of the
                corresponding entry.

        Returns:
            list: mean ordering accuracy

        """
        # group by mol source
        group = defaultdict(list)
        for pred, tgt, m in zip(predictions, targets, mol_sources):
            group[m].append([pred, tgt])

        # analyzer order accuracy for each group
        scores = []
        for _, g in group.items():
            data = np.asarray(g)
            pred = data[:, 0]
            tgt = data[:, 1]
            s = [self.smallest_n_score(pred, tgt, n) for n in range(1, self.max_n + 1)]
            scores.append(s)
        mean_score = np.mean(scores, axis=0)

        return mean_score

    @staticmethod
    def smallest_n_score(prediction, target, n=2):
        """
        Measure how many smallest n elements of source are in that of the target.

        Args:
            prediction (1D array):
            target (1D array):
            n (int): the number of elements to consider

        Returns:
            A float of value {0,1/n, 2/n, ..., n/n}, depending the intersection of the
            smallest n elements between source and target.
        """
        # first n args that will sort the array
        p_args = list(np.argsort(prediction)[:n])
        t_args = list(np.argsort(target)[:n])
        intersection = set(p_args).intersection(set(t_args))
        return len(intersection) / len(p_args)


class EarlyStopping:
    def __init__(self, patience=200, silent=True):
        self.patience = patience
        self.silent = silent
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if not self.silent:
                print("EarlyStopping counter: {}/{}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
