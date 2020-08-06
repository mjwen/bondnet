"""
ResGatedGCN: Residual Gated Graph ConvNets
An Experimental Study of Neural Networks for Variable Graphs
(Xavier Bresson and Thomas Laurent, ICLR 2018)
https://arxiv.org/pdf/1711.07553v2.pdf
"""


import torch
from torch import nn
import logging
from dgl import function as fn
from bondnet.layer.hgatconv import NodeAttentionLayer
from bondnet.layer.utils import LinearN

logger = logging.getLogger(__name__)


class GatedGCNConv(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_fc_layers=1,
        graph_norm=True,
        batch_norm=True,
        activation=nn.ELU(),
        residual=False,
        dropout=0.0,
    ):
        super(GatedGCNConv, self).__init__()
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.activation = activation
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        out_sizes = [output_dim] * num_fc_layers
        acts = [activation] * (num_fc_layers - 1) + [nn.Identity()]
        use_bias = [True] * num_fc_layers
        self.A = LinearN(input_dim, out_sizes, acts, use_bias)
        self.B = LinearN(input_dim, out_sizes, acts, use_bias)
        self.C = LinearN(input_dim, out_sizes, acts, use_bias)
        self.D = LinearN(input_dim, out_sizes, acts, use_bias)
        self.E = LinearN(input_dim, out_sizes, acts, use_bias)
        self.F = LinearN(input_dim, out_sizes, acts, use_bias)
        self.G = LinearN(output_dim, out_sizes, acts, use_bias)
        self.H = LinearN(output_dim, out_sizes, acts, use_bias)
        self.I = LinearN(input_dim, out_sizes, acts, use_bias)

        if self.batch_norm:
            self.bn_node_h = nn.BatchNorm1d(output_dim)
            self.bn_node_e = nn.BatchNorm1d(output_dim)
            self.bn_node_u = nn.BatchNorm1d(output_dim)

        delta = 1e-3
        if dropout >= delta:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    @staticmethod
    def reduce_fn_a2b(nodes):
        """
        Reduce `Eh_j` from atom nodes to bond nodes.

        Expand dim 1 such that every bond has two atoms connecting to it.
        This is to deal with the special case of single atom graph (e.g. H+).
        For such graph, an artificial bond is created and connected to the atom in
        `grapher`. Here, we expand it to let each bond connecting to two atoms.
        This is necessary because, otherwise, the reduce_fn wil not work since
        dimension mismatch.
        """
        x = nodes.mailbox["Eh_j"]
        if x.shape[1] == 1:
            x = x.repeat_interleave(2, dim=1)

        return {"Eh_j": x}

    @staticmethod
    def message_fn(edges):
        return {"Eh_j": edges.src["Eh_j"], "e": edges.src["e"]}

    @staticmethod
    def reduce_fn(nodes):
        Eh_i = nodes.data["Eh"]
        e = nodes.mailbox["e"]
        Eh_j = nodes.mailbox["Eh_j"]

        # TODO select_not_equal is time consuming; it might be improved by passing node
        #  index along with Eh_j and compare the node index to select the different one
        Eh_j = select_not_equal(Eh_j, Eh_i)
        sigma_ij = torch.sigmoid(e)  # sigma_ij = sigmoid(e_ij)

        # (sum_j eta_ij * Ehj)/(sum_j' eta_ij') <= dense attention
        h = torch.sum(sigma_ij * Eh_j, dim=1) / (torch.sum(sigma_ij, dim=1) + 1e-6)

        return {"h": h}

    def forward(self, g, feats, norm_atom=None, norm_bond=None):

        g = g.local_var()

        h = feats["atom"]
        e = feats["bond"]
        u = feats["global"]

        # for residual connection
        h_in = h
        e_in = e
        u_in = u

        g.nodes["atom"].data.update({"Ah": self.A(h), "Dh": self.D(h), "Eh": self.E(h)})
        g.nodes["bond"].data.update({"Be": self.B(e)})
        g.nodes["global"].data.update({"Cu": self.C(u), "Fu": self.F(u)})

        # update bond feature e
        g.multi_update_all(
            {
                "a2b": (fn.copy_u("Ah", "m"), fn.sum("m", "e")),  # A * (h_i + h_j)
                "b2b": (fn.copy_u("Be", "m"), fn.sum("m", "e")),  # B * e_ij
                "g2b": (fn.copy_u("Cu", "m"), fn.sum("m", "e")),  # C * u
            },
            "sum",
        )

        e = g.nodes["bond"].data["e"]
        if self.graph_norm:
            e = e * norm_bond
        if self.batch_norm:
            e = self.bn_node_e(e)
        e = self.activation(e)
        if self.residual:
            e = e_in + e
        g.nodes["bond"].data["e"] = e

        # update atom feature h

        # Copy Eh to bond nodes, without reduction.
        # This is the first arrow in: Eh_j -> bond node -> atom i node
        # The second arrow is done in self.message_fn and self.reduce_fn below
        g.update_all(fn.copy_u("Eh", "Eh_j"), self.reduce_fn_a2b, etype="a2b")

        g.multi_update_all(
            {
                "a2a": (fn.copy_u("Dh", "m"), fn.sum("m", "h")),  # D * h_i
                "b2a": (self.message_fn, self.reduce_fn),  # e_ij [Had] (E * hj)
                "g2a": (fn.copy_u("Fu", "m"), fn.sum("m", "h")),  # F * u
            },
            "sum",
        )

        h = g.nodes["atom"].data["h"]
        if self.graph_norm:
            h = h * norm_atom
        if self.batch_norm:
            h = self.bn_node_h(h)
        h = self.activation(h)
        if self.residual:
            h = h_in + h
        g.nodes["atom"].data["h"] = h

        # update global feature u
        g.nodes["atom"].data.update({"Gh": self.G(h)})
        g.nodes["bond"].data.update({"He": self.H(e)})
        g.nodes["global"].data.update({"Iu": self.I(u)})
        g.multi_update_all(
            {
                "a2g": (fn.copy_u("Gh", "m"), fn.mean("m", "u")),  # G * (mean_i h_i)
                "b2g": (fn.copy_u("He", "m"), fn.mean("m", "u")),  # H * (mean_ij e_ij)
                "g2g": (fn.copy_u("Iu", "m"), fn.sum("m", "u")),  # I * u
            },
            "sum",
        )
        u = g.nodes["global"].data["u"]
        if self.batch_norm:
            u = self.bn_node_u(u)
        u = self.activation(u)
        if self.residual:
            u = u_in + u

        # dropout
        h = self.dropout(h)
        e = self.dropout(e)
        u = self.dropout(u)

        feats = {"atom": h, "bond": e, "global": u}

        return feats


class GatedGCNConv1(GatedGCNConv):
    """
    Compared with GatedGCNConv, we use hgat attention layer to update global feature.  
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_fc_layers=1,
        graph_norm=True,
        batch_norm=True,
        activation=nn.ELU(),
        residual=False,
        dropout=0.0,
    ):
        super().__init__(
            input_dim,
            output_dim,
            num_fc_layers,
            graph_norm,
            batch_norm,
            activation,
            residual,
            dropout,
        )

        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.activation = activation
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        out_sizes = [output_dim] * num_fc_layers
        acts = [activation] * (num_fc_layers - 1) + [nn.Identity()]
        use_bias = [True] * num_fc_layers
        self.A = LinearN(input_dim, out_sizes, acts, use_bias)
        self.B = LinearN(input_dim, out_sizes, acts, use_bias)
        self.C = LinearN(input_dim, out_sizes, acts, use_bias)
        self.D = LinearN(input_dim, out_sizes, acts, use_bias)
        self.E = LinearN(input_dim, out_sizes, acts, use_bias)
        self.F = LinearN(input_dim, out_sizes, acts, use_bias)

        if self.batch_norm:
            self.bn_node_h = nn.BatchNorm1d(output_dim)
            self.bn_node_e = nn.BatchNorm1d(output_dim)
            self.bn_node_u = nn.BatchNorm1d(output_dim)

        delta = 1e-3
        if dropout >= delta:
            self.dropout = nn.Dropout(dropout)
        else:
            logger.warning(f"dropout ({dropout}) smaller than {delta}. Ignored.")
            self.dropout = nn.Identity()

        self.node_attn_layer = NodeAttentionLayer(
            master_node="global",
            attn_nodes=["atom", "bond", "global"],
            attn_edges=["a2g", "b2g", "g2g"],
            in_feats={"atom": output_dim, "bond": output_dim, "global": input_dim},
            out_feats=output_dim,
            num_heads=1,
            num_fc_layers=num_fc_layers,
            feat_drop=0.0,
            attn_drop=0.0,
            negative_slope=0.2,
            residual=False,
            activation=None,
            batch_norm=False,
        )

    def forward(self, g, feats, norm_atom, norm_bond):

        g = g.local_var()

        h = feats["atom"]
        e = feats["bond"]
        u = feats["global"]

        # for residual connection
        h_in = h
        e_in = e
        u_in = u

        g.nodes["atom"].data.update({"Ah": self.A(h), "Dh": self.D(h), "Eh": self.E(h)})
        g.nodes["bond"].data.update({"Be": self.B(e)})
        g.nodes["global"].data.update({"Cu": self.C(u), "Fu": self.F(u)})

        # update bond feature e
        g.multi_update_all(
            {
                "a2b": (fn.copy_u("Ah", "m"), fn.sum("m", "e")),  # A * (h_i + h_j)
                "b2b": (fn.copy_u("Be", "m"), fn.sum("m", "e")),  # B * e_ij
                "g2b": (fn.copy_u("Cu", "m"), fn.sum("m", "e")),  # C * u
            },
            "sum",
        )
        e = g.nodes["bond"].data["e"]
        if self.graph_norm:
            e = e * norm_bond
        if self.batch_norm:
            e = self.bn_node_e(e)
        e = self.activation(e)
        if self.residual:
            e = e_in + e
        g.nodes["bond"].data["e"] = e

        # update atom feature h

        # Copy Eh to bond nodes, without reduction.
        # This is the first arrow in: Eh_j -> bond node -> atom i node
        # The second arrow is done in self.message_fn and self.reduce_fn below
        g.update_all(fn.copy_u("Eh", "Eh_j"), self.reduce_fn_a2b, etype="a2b")

        g.multi_update_all(
            {
                "a2a": (fn.copy_u("Dh", "m"), fn.sum("m", "h")),  # D * h_i
                "b2a": (self.message_fn, self.reduce_fn),  # e_ij [Had] (E * hj)
                "g2a": (fn.copy_u("Fu", "m"), fn.sum("m", "h")),  # F * u
            },
            "sum",
        )
        h = g.nodes["atom"].data["h"]
        if self.graph_norm:
            h = h * norm_atom
        if self.batch_norm:
            h = self.bn_node_h(h)
        h = self.activation(h)
        if self.residual:
            h = h_in + h
        g.nodes["atom"].data["h"] = h

        # update global feature u
        # g.nodes["atom"].data.update({"Gh": self.G(h)})
        # g.nodes["bond"].data.update({"He": self.H(e)})
        # g.nodes["global"].data.update({"Iu": self.I(u)})
        # g.multi_update_all(
        #     {
        #         "a2g": (fn.copy_u("Gh", "m"), fn.mean("m", "u")),  # G * (mean_i h_i)
        #         "b2g": (fn.copy_u("He", "m"), fn.mean("m", "u")),  # H * (mean_ij e_ij)
        #         "g2g": (fn.copy_u("Iu", "m"), fn.sum("m", "u")),  # I * u
        #     },
        #     "sum",
        # )
        # u = g.nodes["global"].data["u"]
        u = self.node_attn_layer(g, u, [h, e, u]).flatten(start_dim=1)
        if self.batch_norm:
            u = self.bn_node_u(u)
        u = self.activation(u)
        if self.residual:
            u = u_in + u

        # dropout
        h = self.dropout(h)
        e = self.dropout(e)
        u = self.dropout(u)

        feats = {"atom": h, "bond": e, "global": u}

        return feats


class GatedGCNConv2(GatedGCNConv):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_fc_layers=1,
        graph_norm=True,
        batch_norm=True,
        activation=nn.ELU(),
        residual=False,
        dropout=0.0,
    ):
        super(GatedGCNConv, self).__init__()
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.activation = activation
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        out_sizes = [output_dim] * num_fc_layers
        acts = [activation] * (num_fc_layers - 1) + [nn.Identity()]
        use_bias = [True] * num_fc_layers
        self.A = LinearN(input_dim, out_sizes, acts, use_bias)
        self.B = LinearN(input_dim, out_sizes, acts, use_bias)
        # self.C = LinearN(input_dim, out_sizes, acts, use_bias)
        self.D = LinearN(input_dim, out_sizes, acts, use_bias)
        self.E = LinearN(input_dim, out_sizes, acts, use_bias)
        # self.F = LinearN(input_dim, out_sizes, acts, use_bias)
        # self.G = LinearN(output_dim, out_sizes, acts, use_bias)
        # self.H = LinearN(output_dim, out_sizes, acts, use_bias)
        # self.I = LinearN(input_dim, out_sizes, acts, use_bias)

        if self.batch_norm:
            self.bn_node_h = nn.BatchNorm1d(output_dim)
            self.bn_node_e = nn.BatchNorm1d(output_dim)
        # self.bn_node_u = nn.BatchNorm1d(output_dim)

        delta = 1e-3
        if dropout >= delta:
            self.dropout = nn.Dropout(dropout)
        else:
            logger.warning(f"dropout ({dropout}) smaller than {delta}. Ignored.")
            self.dropout = nn.Identity()

    def forward(self, g, feats, norm_atom, norm_bond):

        g = g.local_var()

        h = feats["atom"]
        e = feats["bond"]
        # u = feats["global"]

        # for residual connection
        h_in = h
        e_in = e
        # u_in = u

        g.nodes["atom"].data.update({"Ah": self.A(h), "Dh": self.D(h), "Eh": self.E(h)})
        g.nodes["bond"].data.update({"Be": self.B(e)})
        # g.nodes["global"].data.update({"Cu": self.C(u), "Fu": self.F(u)})

        # update bond feature e
        g.multi_update_all(
            {
                "a2b": (fn.copy_u("Ah", "m"), fn.sum("m", "e")),  # A * (h_i + h_j)
                "b2b": (fn.copy_u("Be", "m"), fn.sum("m", "e")),  # B * e_ij
                # "g2b": (fn.copy_u("Cu", "m"), fn.sum("m", "e")),  # C * u
            },
            "sum",
        )
        e = g.nodes["bond"].data["e"]
        if self.graph_norm:
            e = e * norm_bond
        if self.batch_norm:
            e = self.bn_node_e(e)
        e = self.activation(e)
        if self.residual:
            e = e_in + e
        g.nodes["bond"].data["e"] = e

        # update atom feature h

        # Copy Eh to bond nodes, without reduction.
        # This is the first arrow in: Eh_j -> bond node -> atom i node
        # The second arrow is done in self.message_fn and self.reduce_fn below
        g.update_all(fn.copy_u("Eh", "Eh_j"), self.reduce_fn_a2b, etype="a2b")

        g.multi_update_all(
            {
                "a2a": (fn.copy_u("Dh", "m"), fn.sum("m", "h")),  # D * h_i
                "b2a": (self.message_fn, self.reduce_fn),  # e_ij [Had] (E * hj)
                # "g2a": (fn.copy_u("Fu", "m"), fn.sum("m", "h")),  # F * u
            },
            "sum",
        )
        h = g.nodes["atom"].data["h"]
        if self.graph_norm:
            h = h * norm_atom
        if self.batch_norm:
            h = self.bn_node_h(h)
        h = self.activation(h)
        if self.residual:
            h = h_in + h
        g.nodes["atom"].data["h"] = h

        # # update global feature u
        # g.nodes["atom"].data.update({"Gh": self.G(h)})
        # g.nodes["bond"].data.update({"He": self.H(e)})
        # g.nodes["global"].data.update({"Iu": self.I(u)})
        # g.multi_update_all(
        #     {
        #         "a2g": (fn.copy_u("Gh", "m"), fn.mean("m", "u")),  # G * (mean_i h_i)
        #         "b2g": (fn.copy_u("He", "m"), fn.mean("m", "u")),  # H * (mean_ij e_ij)
        #         "g2g": (fn.copy_u("Iu", "m"), fn.sum("m", "u")),  # I * u
        #     },
        #     "sum",
        # )
        # u = g.nodes["global"].data["u"]
        # if self.batch_norm:
        #     u = self.bn_node_u(u)
        # u = self.activation(u)
        # if self.residual:
        #     u = u_in + u

        # dropout
        h = self.dropout(h)
        e = self.dropout(e)
        # u = self.dropout(u)

        # feats = {"atom": h, "bond": e, "global": u}
        feats = {"atom": h, "bond": e}

        return feats


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
    d0, d1, d2, d3 = x.shape
    assert d2 == 2, f"Expect x.shape[2]==2, got {d2}"

    ## method 1, slow
    # rst = []
    # for x1, y1 in zip(x, y):
    #     xx = [x2[0] if not torch.equal(y1, x2[0]) else x2[1] for x2 in x1]
    #     rst.append(torch.stack(xx))
    # rst = torch.stack(rst)

    # method 2, a much faster version
    y = torch.repeat_interleave(y, d1 * d2, dim=0).view(x.shape)
    any_not_equal = torch.any(x != y, dim=3)
    # bool index
    idx1 = any_not_equal[:, :, 0].view(d0, d1, 1)
    idx2 = ~idx1
    idx = torch.cat([idx1, idx2], dim=-1)
    # select result
    rst = x[idx].view(d0, d1, -1)

    return rst
