"""
Heterogeneous Graph Attention Networks on molecule level property.
"""

import torch.nn as nn
from gnn.layer.hgatconv import HGATConv
from gnn.layer.readout import Set2SetThenCat
import warnings
from gnn.utils import warn_stdout


class HGATMol(nn.Module):
    """
    Heterograph attention network.


    Args:
        attn_mechanism (dict of dict): The attention mechanism, i.e. how the node
            features will be updated. The outer dict has `node types` as its key,
            and the inner dict has keys `nodes` and `edges`, where the values (list)
            of `nodes` are the `node types` that the master node will attend to,
            and the corresponding `edges` are the `edge types`.
        attn_order (list): The order to attend the node features. Order matters.
        in_feats (list): input feature size for the corresponding node in `attn_order`.
        num_gat_layers (int): number of graph attention layer
        gat_hidden_size (list): hidden size of graph attention layers
        num_heads (int): number of attention heads, the same for all nodes
        feat_drop (float, optional): [description]. Defaults to 0.0.
        attn_drop (float, optional): [description]. Defaults to 0.0.
        negative_slope (float, optional): [description]. Defaults to 0.2.
        gat_residual (bool, optional): [description]. Defaults to False.
        gat_batch_norm (bool, optional): [description]. Defaults to False.
        gat_activation (torch activation): activation fn of gat layers
        num_fc_layers (int): number of fc layers. Note this is the number of hidden
            layers, i.e. there is an additional fc layer to map feature size to 1.
        fc_hidden_size (list): hidden size of fc layers
        fc_dropout (float): dropout for fc layer
        fc_activation (torch activation): activation fn of fc layers
        fc_drop (float): dropout for fc layer
    """

    def __init__(
        self,
        attn_mechanism,
        attn_order,
        in_feats,
        num_gat_layers=3,
        gat_hidden_size=[32, 64, 128],
        num_heads=4,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        gat_residual=True,
        gat_batch_norm=False,
        gat_activation=nn.ELU(),
        num_lstm_iters=5,
        num_lstm_layers=3,
        set2set_ntypes_direct=["global"],
        num_fc_layers=3,
        fc_hidden_size=[128, 64, 32],
        fc_batch_norm=False,
        fc_activation=nn.ELU(),
        fc_drop=0.0,
    ):
        super(HGATMol, self).__init__()

        # activation fn
        if isinstance(gat_activation, str):
            gat_activation = getattr(nn, gat_activation)()
        if isinstance(fc_activation, str):
            fc_activation = getattr(nn, fc_activation)()

        self.gat_layers = nn.ModuleList()

        # input projection (no dropout, no residual)
        self.gat_layers.append(
            HGATConv(
                attn_mechanism=attn_mechanism,
                attn_order=attn_order,
                in_feats=in_feats,
                out_feats=gat_hidden_size[0],
                num_heads=num_heads,
                feat_drop=0.0,
                attn_drop=0.0,
                negative_slope=negative_slope,
                residual=gat_residual,
                batch_norm=gat_batch_norm,
                activation=gat_activation,
            )
        )

        # hidden gat layers
        for i in range(1, num_gat_layers):
            in_size = [gat_hidden_size[i - 1] * num_heads for _ in in_feats]
            self.gat_layers.append(
                HGATConv(
                    attn_mechanism=attn_mechanism,
                    attn_order=attn_order,
                    in_feats=in_size,
                    out_feats=gat_hidden_size[i],
                    num_heads=num_heads,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    residual=gat_residual,
                    batch_norm=gat_batch_norm,
                    activation=gat_activation,
                )
            )

        ntypes = ["atom", "bond"]
        in_size = [gat_hidden_size[-1] * num_heads for _ in attn_order]

        self.readout_layer = Set2SetThenCat(
            n_iters=num_lstm_iters,
            n_layer=num_lstm_layers,
            ntypes=ntypes,
            in_feats=in_size,
            ntypes_direct_cat=set2set_ntypes_direct,
        )

        # need dropout?
        delta = 1e-3
        if fc_drop < delta:
            warnings.showwarning = warn_stdout
            warnings.warn(
                "`fc_drop = {}` provided for {} smaller than {}. "
                "Ignore dropout.".format(fc_drop, self.__class__.__name__, delta)
            )
            apply_drop = False
        else:
            apply_drop = True

        # for atom and bond feat (# *2 because Set2Set used in Set2SetThenCat has out
        # feature twice the the size  of in feature)
        in_size = (
            gat_hidden_size[-1] * num_heads * 2 + gat_hidden_size[-1] * num_heads * 2
        )
        # for global feat
        if set2set_ntypes_direct is not None:
            in_size += gat_hidden_size[-1] * num_heads

        # fc layer to map to feature to bond energy
        self.fc_layers = nn.ModuleList()
        for i in range(num_fc_layers):
            out_size = fc_hidden_size[i]
            self.fc_layers.append(nn.Linear(in_size, out_size))
            # batch norm
            if fc_batch_norm:
                self.fc_layers.append(nn.BatchNorm1d(out_size))
            # activation
            self.fc_layers.append(fc_activation)
            # dropout
            if apply_drop:
                self.fc_layers.append(nn.Dropout(fc_drop))

            in_size = out_size

        # final output layer, mapping feature to size 1
        self.fc_layers.append(nn.Linear(in_size, 1))

    def forward(self, graph, feats):
        h = feats

        # hgat layer
        for layer in self.gat_layers:
            h = layer(graph, h)

        # readout layer
        h = self.readout_layer(graph, h)

        # fc
        for layer in self.fc_layers:
            h = layer(h)

        return h
