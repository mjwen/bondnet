"""
Heterogeneous Graph Attention Networks on molecule level property.
"""

import torch
import warnings
import torch.nn as nn
from bondnet.layer.hgatconv import HGATConv
from bondnet.layer.readout import Set2SetThenCat
from bondnet.utils import warn_stdout


class HGATMol(nn.Module):
    """
    Heterograph attention network for molecules.


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
        gat_num_fc_layers (int): number of fully-connected layer before attention
        gat_residual (bool, optional): [description]. Defaults to False.
        gat_batch_norm(bool): whether to apply batch norm to gat layer.
        gat_activation (torch activation): activation fn of gat layers
        num_fc_layers (int): number of fc layers. Note this is the number of hidden
            layers, i.e. there is an additional fc layer to map feature size to 1.
        fc_hidden_size (list): hidden size of fc layers
        fc_batch_norm (bool): whether to apply batch norm to fc layer
        fc_activation (torch activation): activation fn of fc layers
        fc_drop (float, optional): dropout ratio for fc layer.
        outdim (int): dimension of the output. For regression, choose 1 and for
            classification, set it to the number of classes.
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
        gat_num_fc_layers=3,
        gat_residual=True,
        gat_batch_norm=False,
        gat_activation="ELU",
        num_lstm_iters=5,
        num_lstm_layers=3,
        set2set_ntypes_direct=["global"],
        num_fc_layers=3,
        fc_hidden_size=[128, 64, 32],
        fc_batch_norm=False,
        fc_activation="ELU",
        fc_drop=0.0,
        outdim=1,
    ):
        super(HGATMol, self).__init__()

        # activation fn
        if isinstance(gat_activation, str):
            self.gat_activation = getattr(nn, gat_activation)()
        if isinstance(fc_activation, str):
            fc_activation = getattr(nn, fc_activation)()

        self.num_heads = num_heads

        self.gat_layers = nn.ModuleList()

        for i in range(num_gat_layers):

            # first layer use not dropout
            if i == 0:
                fd = 0.0
                ad = 0.0
                in_size = in_feats
                act = self.gat_activation
            else:
                fd = feat_drop
                ad = attn_drop
                in_size = [gat_hidden_size[i - 1] * num_heads for _ in in_feats]

                # last layer do not apply activation now, but after average over heads
                # (eq. 6 of the GAT paper) see below in forward()
                if i == num_fc_layers - 1:
                    act = None
                else:
                    act = self.gat_activation

            self.gat_layers.append(
                HGATConv(
                    attn_mechanism=attn_mechanism,
                    attn_order=attn_order,
                    in_feats=in_size,
                    out_feats=gat_hidden_size[i],
                    num_heads=num_heads,
                    num_fc_layers=gat_num_fc_layers,
                    feat_drop=fd,
                    attn_drop=ad,
                    negative_slope=negative_slope,
                    residual=gat_residual,
                    batch_norm=gat_batch_norm,
                    activation=act,
                )
            )

        # set2set readout layer
        ntypes = ["atom", "bond"]
        in_size = [gat_hidden_size[-1] for _ in attn_order]

        self.readout_layer = Set2SetThenCat(
            n_iters=num_lstm_iters,
            n_layer=num_lstm_layers,
            ntypes=ntypes,
            in_feats=in_size,
            ntypes_direct_cat=set2set_ntypes_direct,
        )

        # for atom and bond feat (# *2 because Set2Set used in Set2SetThenCat has out
        # feature twice the the size  of in feature)
        readout_out_size = gat_hidden_size[-1] * 2 + gat_hidden_size[-1] * 2
        # for global feat
        if set2set_ntypes_direct is not None:
            readout_out_size += gat_hidden_size[-1] * len(set2set_ntypes_direct)

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

        # fc layer to map to feature to bond energy
        self.fc_layers = nn.ModuleList()
        in_size = readout_out_size
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

        # final output layer, mapping feature to the corresponding shape
        self.fc_layers.append(nn.Linear(in_size, outdim))

    def forward(self, graph, feats):
        """
        Returns:
            2D tensor: of shape (N, ft_size)
        """

        # hgat layer
        for i, layer in enumerate(self.gat_layers):
            feats = layer(graph, feats)

            # apply activation after average over heads (eq. 6 of the GAT paper)
            # see below in forward()
            if i == len(self.gat_layers) - 1:
                for nt in feats:
                    ft = feats[nt].view(feats[nt].shape[0], self.num_heads, -1)
                    feats[nt] = self.gat_activation(torch.mean(ft, dim=1))

        # readout layer
        feats = self.readout_layer(graph, feats)

        # fc
        for layer in self.fc_layers:
            feats = layer(feats)

        return feats
