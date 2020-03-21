"""Torch modules for GAT for heterograph."""
import torch
from torch import nn
import warnings
from dgl import function as fn
from gnn.utils import warn_stdout


class UnifySize(nn.Module):
    """
    A layer to unify the feature size of nodes of different types.
    Each feature uses a linear fc layer to map the size.

    NOTE, after this transformation, each data point is just a linear combination of its
    feature in the orginal feature space (x_new_ij = x_ik w_kj), there is not mixing of
    feature between data points.

    Args:
        in_feats (dict): feature sizes of nodes with node type as key and size as value
        out_feats (int): output feature size, i.e. the size we will turn all the
            features to
    """

    def __init__(self, in_feats, out_feats):
        super(UnifySize, self).__init__()

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
            dict: size adjusted features
        """
        return {ntype: self.linears[ntype](x) for ntype, x in feats.items()}


class NodeAttentionLayer(nn.Module):
    """
    Graph attention for nodes from other nodes.

    Args:
        master_node (str): node type whose feature to update
        attn_nodes (list of str): node type that the master node attends to
        attn_edges (list of str): type of edge connecting attention nodes to master node;
            should be of same size of `attn_nodes`.
        in_feats (dict): feature sizes with node type as key
        out_feats (int): size of output feature
        num_heads (int): number of heads for multi-head attention
        feat_drop (float, optional): dropout ratio for feature. Defaults to 0.0.
        attn_drop (float, optional): dropout ratio for feature after attention.
            Defaults to 0.0.
        negative_slope (float, optional): negative slope for LeakyRelu. Defaults to 0.2.
        residual (bool, optional): if `True`, apply residual connection for the output.
            Defaults to False.
        activation (torch activation function, optional): activation to apply to the
            output. If `None`, do not apply. Defaults to None.
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
        batch_norm=False,
        activation=None,
    ):

        super(NodeAttentionLayer, self).__init__()
        self.master_node = master_node
        self.attn_nodes = attn_nodes
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.activation = activation
        self.residual = residual
        self.edge_types = [(n, e, master_node) for n, e in zip(attn_nodes, attn_edges)]

        # TODO this could be networks to increase capacity
        # linear FC layer (two purposes, 1. W as in eq1 of the GAT paper. 2. unify size
        # of feature from different node)
        self.fc_layers = nn.ModuleDict(
            {
                nt: nn.Linear(sz, out_feats * num_heads, bias=False)
                for nt, sz in in_feats.items()
            }
        )

        # # linear FC layer (use it only when in size differ from out size)
        # d = {}
        # for nt, sz in in_feats.items():
        #     if sz == out_feats * num_heads:
        #         d[nt] = nn.Identity()
        #     else:
        #         d[nt] = nn.Linear(sz, out_feats * num_heads, bias=False)
        # self.fc_layers = nn.ModuleDict(d)

        # parameters for attention
        self.attn_l = nn.Parameter(torch.zeros(1, num_heads, out_feats))
        self.attn_r = nn.Parameter(torch.zeros(1, num_heads, out_feats))
        self.reset_parameters()
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        # here we use different dropout for each node type
        delta = 1e-3
        if feat_drop >= delta:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            warnings.showwarning = warn_stdout
            warnings.warn(
                "`feat_drop = {}` provided for {} smaller than {}. "
                "Ignore dropout.".format(feat_drop, self.__class__.__name__, delta)
            )
            self.feat_drop = nn.Identity()

        # attn_drop is used for edge_types, not node types, so here we create dropout
        # using attn_nodes instead of in_feats as for feat_drop
        if attn_drop >= delta:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            warnings.showwarning = warn_stdout
            warnings.warn(
                "`attn_drop = {}` provided for {} smaller than {}. "
                "Ignore dropout.".format(attn_drop, self.__class__.__name__, delta)
            )
            self.attn_drop = nn.Identity()

        # batch normalization
        if batch_norm:
            self.batch_norm_layer = nn.BatchNorm1d(num_features=out_feats * num_heads)
        else:
            self.batch_norm_layer = nn.Identity()

    def reset_parameters(self):
        """Reinitialize parameters."""
        gain = nn.init.calculate_gain("relu")
        # for nt, layer in self.fc_layers.items():
        #     nn.init.xavier_normal_(layer.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

    def forward(self, graph, master_feats, attn_feats):
        """
        Args:
            graph (DGLHeteroGraph or BatchedDGLHeteroGraph): The graph.
            master_feats (torch.Tensor): Feature of the master node that
                will be updated. It is of shape :math:`(N, D_{in})` where :math:`N` is
                the number of nodes and :math:`D_{in}` is the feature size.
            attn_feats (list of torch.Tensor): Features of the attention nodes.
                Each element is of shape :math:`(N, D_{in})` where :math:`N` is the
                number of nodes and :math:`D_{in}` is the feature size.
                Note that the order of the features should corresponds to that of the
                `attn_nodes` provided at instantiation.

        Returns:
            torch.Tensor: The updated master feature of shape :math:`(N, H, D_{out})`
            where :math:`N` is the number of nodes, :math:`H` is the number of heads,
            and :math:`D_{out}` is the size of the output master feature.
        """
        graph = graph.local_var()

        # assign data
        # master node
        master_feats = self.feat_drop(master_feats)  # (N, in)
        master_feats = self.fc_layers[self.master_node](master_feats).view(
            -1, self.num_heads, self.out_feats
        )  # (N, H, out)
        er = (master_feats * self.attn_r).sum(dim=-1).unsqueeze(-1)  # (N, H, 1)
        graph.nodes[self.master_node].data.update({"ft": master_feats, "er": er})

        # attention node
        for ntype, feats in zip(self.attn_nodes, attn_feats):
            feats = self.feat_drop(feats)
            feats = self.fc_layers[ntype](feats).view(-1, self.num_heads, self.out_feats)
            el = (feats * self.attn_l).sum(dim=-1).unsqueeze(-1)
            graph.nodes[ntype].data.update({"ft": feats, "el": el})

        # compute edge attention
        e = []  # each component is of shape(Ne, H, 1)
        for etype in self.edge_types:
            graph.apply_edges(fn.u_add_v("el", "er", "e"), etype=etype)
            e.append(self.leaky_relu(graph.edges[etype].data.pop("e")))

        # softmax, each component is of shape(Ne, H, 1)
        softmax = heterograph_edge_softmax(graph, self.edge_types, e)

        # apply attention dropout
        for etype, a in zip(self.edge_types, softmax):
            graph.edges[etype].data["a"] = self.attn_drop(a)

        # message passing, "ft" is of shape(H, out), and "a" is of shape(H, 1)
        # computing the part inside the parenthesis of eq. 4 of the GAT paper
        graph.multi_update_all(
            {
                etype: (fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
                for etype in self.edge_types
            },
            "sum",
        )
        rst = graph.nodes[self.master_node].data["ft"]  # shape(N, H, out)

        # residual
        if self.residual:
            rst = rst + master_feats

        # batch normalization
        rst = rst.view(-1, self.num_heads * self.out_feats)
        rst = self.batch_norm_layer(rst)
        rst = rst.view(-1, self.num_heads, self.out_feats)

        # activation
        if self.activation:
            rst = self.activation(rst)

        return rst


class HGATConv(nn.Module):
    """
    Graph attention convolution layer for hetero graph that attends between different
    (and the same) type of nodes.

    Args:
        attn_mechanism (dict of dict): The attention mechanism, i.e. how the node
            features will be updated. The outer dict has `node types` as its key
            and the inner dict has keys `nodes` and `edges`, where the values (list)
            of `nodes` are the `node types` that the master node will attend to,
            and the corresponding `edges` are the `edge types`.
        attn_order (list): `node type` string that specify the order to attend the node
            features.
        in_feats (list): input feature size for the corresponding (w.r.t. index) node
            in `attn_order`.
        out_feats (int): output feature size, the same for all nodes
        num_heads (int): number of attention heads, the same for all nodes
        feat_drop (float, optional): [description]. Defaults to 0.0.
        attn_drop (float, optional): [description]. Defaults to 0.0.
        negative_slope (float, optional): [description]. Defaults to 0.2.
        residual (bool, optional): [description]. Defaults to False.
        batch_norm(bool): whether to apply batch norm to the output
        activation (nn.Moldule or str): activation fn
    """

    def __init__(
        self,
        attn_mechanism,
        attn_order,
        in_feats,
        out_feats,
        num_heads=4,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        batch_norm=False,
        activation=None,
    ):

        super(HGATConv, self).__init__()

        self.attn_mechanism = attn_mechanism
        self.master_nodes = attn_order

        in_feats_map = dict(zip(attn_order, in_feats))

        self.layers = nn.ModuleDict()

        for ntype in self.master_nodes:
            self.layers[ntype] = NodeAttentionLayer(
                master_node=ntype,
                attn_nodes=self.attn_mechanism[ntype]["nodes"],
                attn_edges=self.attn_mechanism[ntype]["edges"],
                in_feats=in_feats_map,
                out_feats=out_feats,
                num_heads=num_heads,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                activation=activation,
                batch_norm=batch_norm,
            )
            in_feats_map[ntype] = num_heads * out_feats

    def forward(self, graph, feats):
        """
        Args:
            graph (dgl heterograph): the graph
            feats (dict): node features with node type as key and the corresponding
            features as value.

        Returns:
            dict: updated node features with the same keys as in `feats`.
                Each feature value has a shape of `(N, out_feats*num_heads)`, where
                `N` is the number of nodes (different for different key) and
                `out_feats` and `num_heads` are the out feature size and number
                of heads specified at instantiation (the same for different keys).
        """
        updated_feats = {k: v for k, v in feats.items()}
        for ntype in self.master_nodes:
            master_feats = updated_feats[ntype]
            attn_feats = [updated_feats[t] for t in self.attn_mechanism[ntype]["nodes"]]
            ft = self.layers[ntype](graph, master_feats, attn_feats)
            updated_feats[ntype] = ft.flatten(start_dim=1)  # flatten the head dimension
        return updated_feats


def heterograph_edge_softmax(graph, edge_types, edge_data):
    r"""Edge softmax for heterograph.

     For a node :math:`i`, edge softmax is an operation of computing

    .. math::
      a_{ij} = \frac{\exp(z_{ij})}{\sum_{j\in\mathcal{N}(i)}\exp(z_{ij})}

    where :math:`z_{ij}` is a signal of edge :math:`j\rightarrow i`, also called logits
    in the context of softmax. :math:`\mathcal{N}(i)` is the set of nodes that have an
    edge to :math:`i`. The type of j is ignored, i.e. it runs over all j that directs
    to i, no matter what the node type of j is.

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

    # assign data
    max_e = []
    min_e = []
    for etype, edata in zip(edge_types, edge_data):
        g.edges[etype].data["e"] = edata
        max_e.append(torch.max(edata))
        min_e.append(torch.min(edata))
    max_e = max(max_e)
    min_e = min(min_e)

    # The softmax trick, making the exponential stable.
    # see https://stackoverflow.com/questions/42599498/numercially-stable-softmax
    # max_e > 64 to prevent overflow; min_e<-64 to prevent underflow
    #
    # Of course, we can apply the trick all the time, but here we choose to apply only
    # in some conditions to save some time, since multi_update_all is really expensive.
    if max_e > 64.0 or min_e < -64.0:
        # e max (fn.max operates on the axis of features from different nodes)
        g.multi_update_all(
            {etype: (fn.copy_e("e", "m"), fn.max("m", "emax")) for etype in edge_types},
            "max",
        )
        # subtract max and compute exponential
        for etype in edge_types:
            g.apply_edges(fn.e_sub_v("e", "emax", "e"), etype=etype)

    for etype in edge_types:
        g.edges[etype].data["out"] = torch.exp(g.edges[etype].data["e"])

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
