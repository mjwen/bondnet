"""Torch modules for GAT for heterograph."""
# pylint: disable=no-member
from functools import partial
import torch
from torch import nn
from dgl import function as fn


class UnifySize(nn.Module):
    """
    A layer to to unify the feature size of different types nodes.

    Args:
        in_feats (dict): in feature sizes of nodes with node type as key and size as value
        out_feats (int): output feature size
    """

    def __init__(self, in_feats, out_feats):
        super(UnifySize, self).__init__()
        # NOTE, we could use nolinear mapping here to add more power
        self.linears = nn.ModuleDict(
            {
                ntype: nn.Linear(size, out_feats, bias=False)
                for ntype, size in in_feats.items()
            }
        )

    def forward(self, feats):
        """
        Args:
            feats (dict): features dict with node type as key and feature as value

        Returns:
            dict: size adjusted features dict
        """
        return {ntype: self.linears[ntype](x) for ntype, x in feats.items()}


class NodeAttentionLayer(nn.Module):
    """
    Graph attention for nodes from other nodes.
    """

    def __init__(
        self,
        master_node,
        attn_nodes,
        attn_edges,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
    ):
        """
        Args:
            master_node (str): node type to update its attributes.
            attn_nodes (list of str): node types that the master node will
                attention to.
            attn_edges (list of str): the type of the edges that connect
                attention nodes to master nodes, should be of same size of attention
                nodes
            in_feats (int): [description]
            out_feats (int): [description]
            num_heads (int): [description]
            feat_drop (float, optional): [description]. Defaults to 0.0.
            attn_drop (float, optional): [description]. Defaults to 0.0.
            negative_slope (float, optional): [description]. Defaults to 0.2.
            residual (bool, optional): [description]. Defaults to False.
            activation ([type], optional): [description]. Defaults to None.

        Returns:
            torch.Tensor: The output feature of shape :matorch:`(N, H, D_{out})` where
                :matorch:`H` is torche number of heads, and :matorch:`D_{out}` is size of
                output feature.
        """
        super(NodeAttentionLayer, self).__init__()
        self.master_node = master_node
        self.attn_nodes = attn_nodes
        self.attn_edges = attn_edges
        self.num_heads = num_heads
        self.in_feats = in_feats
        self.out_feats = out_feats

        # NOTE we may want to combine this fc layer with the UnifySize layer, if we want
        # to use smaller number of parameters. Of course, this depents on the actural
        # size of in_feats and out_feats and num_heats
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        # NOTE dropout could also be considered separately for different types of nodes
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(in_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self.activation = activation

        self.edge_types = [(n, e, master_node) for n, e in zip(attn_nodes, attn_edges)]

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        # NOTE do we need other type of initialization, e.g. Kaiming?
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, master_feats, attn_feats):
        """Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLHeteroGraph
            The graph.
        master_feats : torch.Tensor
            The input feature of the master node, which attributes will be updated.
            The input feature of shape :matorch:`(N, D_{in})` where :matorch:`D_{in}`
            is size of input feature, :matorch:`N` is torche number of nodes.
        attn_feats : list of torch.Tensor
            The input features of the attention nodes.
            The input feature of shape :matorch:`(N, D_{in})` where :matorch:`D_{in}`
            is size of input feature, :matorch:`N` is torche number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :matorch:`(N, H, D_{out})` where :matorch:`H`
            is torche number of heads, and :matorch:`D_{out}` is size of output feature.
        """
        graph = graph.local_var()

        # assign data
        # master node
        master_h = self.feat_drop(master_feats)  # shape(N, in)
        feats = self.fc(master_h).view(
            -1, self.num_heads, self.out_feats
        )  # shape(N, H, out)
        er = (feats * self.attn_r).sum(dim=-1).unsqueeze(-1)  # shape(N, H, 1)
        graph.nodes[self.master_node].data.update({"ft": feats, "er": er})
        # attention node
        for node, feats in zip(self.attn_nodes, attn_feats):
            h = self.feat_drop(feats)
            feats = self.fc(h).view(-1, self.num_heads, self.out_feats)
            el = (feats * self.attn_l).sum(dim=-1).unsqueeze(-1)
            graph.nodes[node].data.update({"ft": feats, "el": el})

        # compute edge attention
        e = []  # each componenet is of shape(Ne, H, 1)
        for etype in self.edge_types:
            graph.apply_edges(fn.u_add_v("el", "er", "e"), etype=etype)
            e.append(self.leaky_relu(graph.edges[etype].data.pop("e")))

        # softmax
        softmax = heterograph_edge_softmax(
            graph, self.master_node, self.attn_nodes, self.attn_edges, e
        )  # each componenet is of shape(Ne, H, 1)
        for etype, a in zip(self.edge_types, softmax):
            graph.edges[etype].data["a"] = self.attn_drop(a)

        # message passing  ("ft" is of shape(H, out), and "a" is of shape(H, 1))
        graph.multi_update_all(
            {
                etype: (fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
                for etype in self.edge_types
            },
            "sum",
        )
        rst = graph.nodes[self.master_node].data["ft"]  # shape(N, H, out)

        # residual
        if self.res_fc is not None:
            resval = self.res_fc(master_h).view(master_h.shape[0], -1, self.out_feats)
            rst = rst + resval

        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst


class HGATConv(nn.Module):
    """
    Graph attention convolution layer for heterograph that attends between different
    (and the same) type of nodes.
    """

    def __init__(
        self,
        attn_mechanism,
        attn_order,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        unify_size=False,
    ):
        """
        Args:
            nn ([type]): [description]
            in_feats (list): list of input feature size for the corresponding (w.r.t.
                index) node in `master_nodes`
            out_feats (int): output feature size, the same for all nodes
            num_heads (int): number of attnetion heads, the same for all nodes
            master_nodes (list, optional): type of the nodes whose features will be updated.
                Update proceeds in the order of this list. Defaults to ["atom", "bond",
                "global"].
            attn_nodes (list, optional): type of the nodes that attend to the corresponding
                master node (order matters). Defaults to [["bond", "global"], ["atom",
                "global"], ["atom", "bond"]].
            attn_edges (list, optional): type of the edges directs from the attention node
                to the corresponding master node (order matters). Defaults to [["b2a",
                "g2a", ["a2b", "g2b"], ["a2g", "b2g"]]].
            feat_drop (float, optional): [description]. Defaults to 0.0.
            attn_drop (float, optional): [description]. Defaults to 0.0.
            negative_slope (float, optional): [description]. Defaults to 0.2.
            residual (bool, optional): [description]. Defaults to False.
            unify_size (bool, optional): [description]. Defaults to False.
        """
        super(HGATConv, self).__init__()

        self.attn_mechamism = attn_mechanism
        self.master_nodes = attn_order

        self.layers = nn.ModuleDict()

        # unify size layer
        if unify_size:
            self.layers["unify_size"] = UnifySize(
                {n: sz for n, sz in zip(self.master_nodes, in_feats)}, out_feats
            )
            size_first = out_feats
        else:
            size_first = list(set(in_feats))
            msg = (
                "'in_feats = {}': elements not equal. Either set them to be the same or "
                "unify_size' to be 'True'".format(in_feats)
            )
            assert len(size_first) == 1, msg
            size_first = size_first[0]

        for i, ntype in enumerate(self.master_nodes):
            if i == 0:
                in_size = size_first
            else:
                # in_size = out_feats * num_heads  # output size of the previous layer
                in_size = out_feats

            # NOTE partial is used as a readout function to reduce the heads dimenstion
            activation = partial(torch.mean, dim=1)
            self.layers[ntype] = NodeAttentionLayer(
                ntype,
                self.attn_mechamism[ntype]["nodes"],
                self.attn_mechamism[ntype]["edges"],
                in_size,
                out_feats,
                num_heads,
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                activation,
            )

    def forward(self, g, feats):
        """
        Args:
            g (dgl heterograph): a heterograph
            feats (dict): node features with node type as key and the corresponding
            features as value.

        Returns:
            dict: updated node features with the same keys in `feats`. Each feature
            value has a shape of `(N, out_feats*num_heads)`, where `N` is the number of
            nodes (different for different key) and `out_feats` and `num_heads` are the
            out feature size and number of heads specified at instantiation (the same
            for different keys.)
        """
        if "unify_size" in self.layers:
            updated_feats = self.layers["unify_size"](feats)
        else:
            updated_feats = {k: v for k, v in feats.items()}
        for ntype in self.master_nodes:
            master_feats = updated_feats[ntype]
            attn_feats = [updated_feats[t] for t in self.attn_mechamism[ntype]["nodes"]]
            ft = self.layers[ntype](g, master_feats, attn_feats)
            updated_feats[ntype] = ft.flatten(start_dim=1)  # flatten the head dimension
        return updated_feats


def get_edge_types(master_node, attn_nodes, attn_edges):
    return [(n, e, master_node) for n, e in zip(attn_nodes, attn_edges)]


def heterograph_edge_softmax(graph, master_node, attn_nodes, attn_edges, edge_data):
    r"""Edge softmax for heterograph.

     For a node :math:`i`, edge softmax is an operation of computing

    .. math::
      a_{ij} = \frac{\exp(z_{ij})}{\sum_{j\in\mathcal{N}(i)}\exp(z_{ij})}

    where :math:`z_{ij}` is a signal of edge :math:`j\rightarrow i`, also called logits
    in the context of softmax. :math:`\mathcal{N}(i)` is the set of nodes that have an
    edge to :math:`i`. The type of j is ignored, i.e. it runs over all j that directs
    to i, no matter what the node type of j is.

    Pseudo-code:

    .. code:: python

        score = dgl.EData(g, score)
        score_max = score.dst_max()  # of type dgl.NData
        score = score - score_max  # edge_sub_dst, ret dgl.EData
        score_sum = score.dst_sum()  # of type dgl.NData
        out = score / score_sum    # edge_div_dst, ret dgl.EData
        return out.data
    ""[summary]

    Returns:
        [type]: [description]
    """
    g = graph.local_var()
    edge_types = get_edge_types(master_node, attn_nodes, attn_edges)

    # assign data
    for etype, edata in zip(edge_types, edge_data):
        g.edges[etype].data["e"] = edata

    # e max (fn.max operates on the axis of features from different nodes)
    g.multi_update_all(
        {etype: (fn.copy_e("e", "m"), fn.max("m", "emax")) for etype in edge_types}, "max"
    )

    # subtract max. The softmax trick, see
    # https://jamesmccaffrey.wordpress.com/2016/03/04/the-max-trick-when-computing-softmax
    for etype in edge_types:
        g.apply_edges(fn.e_sub_v("e", "emax", "out"), etype=etype)
        g.edges[etype].data["out"] = torch.exp(g.edges[etype].data["out"])

    # e sum
    g.multi_update_all(
        {etype: (fn.copy_e("out", "m"), fn.sum("m", "out_sum")) for etype in edge_types},
        "sum",
    )

    a = []
    for etype in edge_types:
        g.apply_edges(fn.e_div_v("out", "out_sum", "a"), etype=etype)
        a.append(g.edges[etype].data["a"])
    return a
