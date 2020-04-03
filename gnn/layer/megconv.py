"""Torch modules for GAT for heterograph."""
import torch
from torch import nn
import warnings
from dgl import function as fn
from gnn.utils import warn_stdout


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


class BondUpdateLayer(nn.Module):
    def __init__(
        self,
        master_node,
        attn_nodes,
        attn_edges,
        in_feats,
        out_feats,
        num_heads,
        num_fc_layers=3,
        fc_activation=nn.Softplus(),
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        batch_norm=False,
        activation=None,
    ):

        super(BondUpdateLayer, self).__init__()
        self.master_node = master_node
        self.attn_nodes = attn_nodes
        self.edge_types = [(n, e, master_node) for n, e in zip(attn_nodes, attn_edges)]

        in_size = in_feats["bond"] + in_feats["atom"] * 2 + in_feats["global"]
        # out_sizes = [out_feats] * num_fc_layers
        out_sizes = [64, 64, 32]
        act = [fc_activation] * (num_fc_layers - 1) + [nn.Identity()]
        use_bias = [True] * num_fc_layers
        self.fc_layers = LinearN(in_size, out_sizes, act, use_bias)

    def forward(self, graph, master_feats, attn_feats):
        graph = graph.local_var()

        # assign feats
        graph.nodes[self.master_node].data.update({"ft": master_feats})
        for ntype, feats in zip(self.attn_nodes, attn_feats):
            graph.nodes[ntype].data.update({"ft": feats})

        for etype in self.edge_types:
            graph.update_all(
                fn.copy_u("ft", "m"), self.reduce_fn, self.apply_node_fn, etype=etype
            )

        feats = self.fc_layers(graph.nodes[self.master_node].data["ft"])

        return feats

    @staticmethod
    def reduce_fn(nodes):
        return {"m": torch.flatten(nodes.mailbox["m"], start_dim=1)}

    @staticmethod
    def apply_node_fn(nodes):
        return {"ft": torch.cat([nodes.data["ft"], nodes.data["m"]], dim=1)}


class AtomUpdateLayer(nn.Module):
    def __init__(
        self,
        master_node,
        attn_nodes,
        attn_edges,
        in_feats,
        out_feats,
        num_heads,
        num_fc_layers=3,
        fc_activation=nn.Softplus(),
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        batch_norm=False,
        activation=None,
    ):

        super(AtomUpdateLayer, self).__init__()
        self.master_node = master_node
        self.attn_nodes = attn_nodes
        self.edge_types = [(n, e, master_node) for n, e in zip(attn_nodes, attn_edges)]

        in_size = in_feats["atom"] + in_feats["bond"] + in_feats["global"]
        # out_sizes = [out_feats] * num_fc_layers
        out_sizes = [64, 64, 32]
        act = [fc_activation] * (num_fc_layers - 1) + [nn.Identity()]
        use_bias = [True] * num_fc_layers
        self.fc_layers = LinearN(in_size, out_sizes, act, use_bias)

    def forward(self, graph, master_feats, attn_feats):
        graph = graph.local_var()

        # assign feats
        graph.nodes[self.master_node].data.update({"ft": master_feats})
        for ntype, feats in zip(self.attn_nodes, attn_feats):
            graph.nodes[ntype].data.update({"ft": feats})

        for etype in self.edge_types:
            graph.update_all(
                fn.copy_u("ft", "m"), fn.mean("m", "m"), self.apply_node_fn, etype=etype
            )

        feats = self.fc_layers(graph.nodes[self.master_node].data["ft"])

        x = feats.shape
        return feats

    @staticmethod
    def apply_node_fn(nodes):
        shape1 = nodes.data["ft"].shape
        shape2 = nodes.data["m"].shape
        ft = torch.cat([nodes.data["ft"], nodes.data["m"]], dim=1)
        shape3 = ft.shape
        return {"ft": ft}


GlobalUpdateLayer = AtomUpdateLayer


class MEGConv(nn.Module):
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
        num_fc_layers (int): number of fully-connected layer before attention
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
        num_fc_layers=3,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        batch_norm=False,
        activation=None,
        first_block=False,
    ):

        super(MEGConv, self).__init__()

        self.attn_mechanism = attn_mechanism
        self.master_nodes = attn_order

        self.residual = residual

        in_feats_map = dict(zip(attn_order, in_feats))

        # linear fc
        self.linear_fc = nn.ModuleDict()
        for ntype in self.master_nodes:
            if first_block:
                in_size = in_feats_map[ntype]
                out_sizes = [64, 32]
                act = [activation] * 2
                use_bias = [True] * 2
                self.linear_fc[ntype] = LinearN(in_size, out_sizes, act, use_bias)
            else:
                self.linear_fc[ntype] = nn.Identity()

            in_size = {k: 32 for k in in_feats_map}

        self.layers = nn.ModuleDict()
        for ntype in self.master_nodes:
            if ntype == "bond":
                self.layers[ntype] = BondUpdateLayer(
                    master_node=ntype,
                    attn_nodes=self.attn_mechanism[ntype]["nodes"],
                    attn_edges=self.attn_mechanism[ntype]["edges"],
                    in_feats=in_size,
                    out_feats=out_feats,
                    num_heads=num_heads,
                    num_fc_layers=num_fc_layers,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    residual=residual,
                    activation=activation,
                    batch_norm=batch_norm,
                )
            elif ntype == "atom":
                self.layers[ntype] = AtomUpdateLayer(
                    master_node=ntype,
                    attn_nodes=self.attn_mechanism[ntype]["nodes"],
                    attn_edges=self.attn_mechanism[ntype]["edges"],
                    in_feats=in_size,
                    out_feats=out_feats,
                    num_heads=num_heads,
                    num_fc_layers=num_fc_layers,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    residual=residual,
                    activation=activation,
                    batch_norm=batch_norm,
                )
            elif ntype == "global":
                self.layers[ntype] = GlobalUpdateLayer(
                    master_node=ntype,
                    attn_nodes=self.attn_mechanism[ntype]["nodes"],
                    attn_edges=self.attn_mechanism[ntype]["edges"],
                    in_feats=in_size,
                    out_feats=out_feats,
                    num_heads=num_heads,
                    num_fc_layers=num_fc_layers,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    residual=residual,
                    activation=activation,
                    batch_norm=batch_norm,
                )

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
        feats_in = {k: v for k, v in feats.items()}

        feats_linear_fc = dict()
        for ntype in self.master_nodes:
            feats_linear_fc[ntype] = self.linear_fc[ntype](feats_in[ntype])

        updated_feats = {k: v for k, v in feats_linear_fc.items()}
        for ntype in self.master_nodes:
            master_feats = updated_feats[ntype]
            attn_feats = [updated_feats[t] for t in self.attn_mechanism[ntype]["nodes"]]
            updated_feats[ntype] = self.layers[ntype](graph, master_feats, attn_feats)

        # residual
        if self.residual:
            for k in updated_feats:
                x = updated_feats[k].shape
                y = feats_linear_fc[k].shape
                updated_feats[k] += feats_linear_fc[k]

        return updated_feats
