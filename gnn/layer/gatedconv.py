"""
ResGatedGCN: Residual Gated Graph ConvNets
An Experimental Study of Neural Networks for Variable Graphs
(Xavier Bresson and Thomas Laurent, ICLR 2018)
https://arxiv.org/pdf/1711.07553v2.pdf
"""


import torch
from torch import nn
from dgl import function as fn
import warnings


class LinearN(nn.Module):
    """
    N stacked linear layers.

    Args:
        in_size (int): input feature size
        out_sizes (list): size of each layer
        activations (list): activation function of each layer
        use_bias (list): whether to use bias for the linear layer
    """

    def __init__(self, in_size, out_sizes, activations, use_bias):
        super(LinearN, self).__init__()

        self.fc_layers = nn.ModuleList()
        for out, act, b in zip(out_sizes, activations, use_bias):
            self.fc_layers.append(nn.Linear(in_size, out, bias=b))
            self.fc_layers.append(act)
            in_size = out

    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)
        return x


class GatedGCNLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        graph_norm=True,
        batch_norm=True,
        activation=nn.ELU(),
        residual=False,
        dropout=0.0,
    ):
        super().__init__()
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.activation = activation
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.F = nn.Linear(input_dim, output_dim, bias=True)
        self.G = nn.Linear(input_dim, output_dim, bias=True)
        self.H = nn.Linear(input_dim, output_dim, bias=True)
        self.I = nn.Linear(input_dim, output_dim, bias=True)

        if self.batch_norm:
            self.bn_node_h = nn.BatchNorm1d(output_dim)
            self.bn_node_e = nn.BatchNorm1d(output_dim)
            self.bn_node_u = nn.BatchNorm1d(output_dim)

        delta = 1e-3
        if dropout >= delta:
            self.dropout = nn.Dropout(dropout)
        else:
            warnings.warn(f"dropout ({dropout}) smaller than {delta}. Ignored.")
            self.dropout = nn.Identity()

    def message_fn(self, edges):
        Eh_j = edges.src["Eh"]
        e = edges.data["e"]

        return {"Eh_j": Eh_j, "e": e}

    def reduce_func(self, nodes):
        Eh_i = nodes.data["Eh"]
        e = nodes.mailbox["e"]
        Eh_j = nodes.mailbox["Eh_j"]

        Eh_j = select_not_equal(Eh_j, Eh_i)

        sigma_ij = torch.sigmoid(e)  # sigma_ij = sigmoid(e_ij)

        # (sum_j eta_ij * Ehj)/(sum_j' eta_ij') <= dense attention
        h = torch.sum(sigma_ij * Eh_j, dim=1) / (torch.sum(sigma_ij, dim=1) + 1e-6)

        return {"h": h}

    def forward(self, g, feats, snorm_n=None, snorm_e=None):

        g = g.local_vars()

        h = feats["atom"]
        e = feats["bond"]
        u = feats["global"]

        # for residual connection
        h_in = h
        e_in = e
        u_in = u

        g.nodes["atom"].data.update(
            {"Ah": self.A(h), "Dh": self.D(h), "Ee": self.E(h), "Gh": self.G(h)}
        )
        g.nodes["bond"].data.update({"Be": self.B(e), "He": self.H(e)})
        g.nodes["global"].data.update({"Cu": self.C(u), "Fu": self.F(u), "Iu": self.I(u)})

        # update bond feature e
        g.multi_update_all(
            {
                "a2b": (fn.copy_u("Ah", "m"), fn.sum("m", "e")),  # A * (h_i + h_j)
                "b2b": (fn.copy_u("Be", "m"), fn.sum("m", "e")),  # B * e_ij
                "g2b": (fn.copy_u("Cu", "m"), fn.sum("m", "e")),  # C * u
            },
            "sum",
        )

        # update atom feature h

        # Copy Eh to bond nodes, without reduction.
        # This is the first arrow in: Eh_j -> bond node -> atom i node
        # The second arrow is done in self.message_fn and self.reduce_fn below
        g.update_all(
            fn.copy_u("Eh", "Eh"), lambda nodes: {"Eh", nodes.mailbox["Eh"]}, etype="a2b"
        )

        g.multi_update_all(
            {
                "a2a": (fn.copy_u("Dh", "m"), fn.sum("m", "h")),  # D * h_i
                "b2a": (self.message_fn, self.reduce_fn),  # e_ij [Had] (E * hj)
                "g2a": (fn.copy_u("Fu", "m"), fn.sum("m", "h")),  # F * u
            },
            "sum",
        )

        # update global feature u
        g.multi_update_all(
            {
                "a2g": (fn.copy_u("Gh", "m"), fn.mean("m", "u")),  # G * (mean_i h_i)
                "b2g": (fn.copy_u("He", "m"), fn.mean("m", "u")),  # H * (mean_ij e_ij)
                "g2g": (fn.copy_u("Iu", "m"), fn.sum("m", "u")),  # I * u
            },
            "sum",
        )

        # result of graph convolution
        h = g.nodes["atom"].data["h"]
        e = g.nodes["bond"].data["e"]
        u = g.nodes["global"].data["u"]

        # normalize activation w.r.t. graph size
        if self.graph_norm:
            h = h * snorm_n
            e = e * snorm_e

        # batch normalization
        if self.batch_norm:
            h = self.bn_node_h(h)
            e = self.bn_node_e(e)
            u = self.bn_node_u(u)

        h = self.actiation(h)
        e = self.activation(e)
        u = self.actiation(u)

        # residual connection
        if self.residual:
            h = h_in + h
            e = e_in + e
            u = u_in + u

        # dropout
        h = self.dropout(h)
        e = self.dropout(e)
        u = self.dropout(u)

        feats = {"atom": h, "bond": e, "global": u}

        return feats

    # def __repr__(self):
    #     return "{}(in_channels={}, out_channels={})".format(
    #         self.__class__.__name__, self.in_channels, self.out_channels
    #     )


def select_not_equal(x, y):
    """Subselect an array from x, which is not equal to the corresponding element
    in y.

    Args:
        x (4D tensor): shape are d0: node batch dim, d1: number of edges dim,
        d2: selection dim, d3: feature dim
        y (2D tensor): shape are 0: nodes batch dim, 1: feature dim

    For example:
    >>> x =[[ [ [0,1,2],
    ...         [3,4,5] ],
    ...       [ [0,1,2],
    ...         [6,7,8] ]
    ...     ],
    ...     [ [ [0,1,2],
    ...         [3,4,5] ],
    ...       [ [3,4,5],
    ...         [6,7,8] ]
    ...     ]
    ...    ]
    >>>
    >>> y = [[0,1,2],
    ...      [3,4,5]]
    >>>
    >>> select_no_equal(x,y)
    ... [[[3,4,5],
    ...   [6,7,8]],
    ...  [[0,1,2],
    ...   [6,7,8]]
    Returns:
        3D tensor: of shape (d0, d1, d3)

    """
    assert x.shape[2] == 2, f"Expect x.shape[2]==2, got {x.shape[2]}"

    rst = []
    for x1, y1 in zip(x, y):
        xx = [x2[0] if not torch.equal(y1, x2[0]) else x2[1] for x2 in x1]
        rst.append(torch.stack(xx))
    rst = torch.stack(rst)

    return rst
