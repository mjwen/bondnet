import torch
from torch import nn
import warnings


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
