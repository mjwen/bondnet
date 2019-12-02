"""
Graph Attention Networks for heterograph.
"""

import torch.nn as nn
from gnn.layer.hgatconv import HGATConv
from gnn.layer.readout import ConcatenateMeanMax


class HGAT(nn.Module):
    """
    Heterograph attention network.

    Args:
        nn ([type]): [description]
        g ([type]): [description]
        num_heads (int, optional): [description]. Defaults to 8.
        hidden_size (int, optional): [description]. Defaults to 64.
        num_layers (int, optional): [description]. Defaults to 2.
        feat_drop (float, optional): [description]. Defaults to 0.0.
        attn_drop (float, optional): [description]. Defaults to 0.0.
        negative_slope (float, optional): [description]. Defaults to 0.2.
        residual (bool, optional): [description]. Defaults to False.
        activation ([type], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        attn_mechanism,
        attn_order,
        in_feats,
        gat_hidden_size=64,
        num_gat_layers=3,
        num_heads=8,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        num_fc_layers=3,
        fc_hidden_size=64,
        fc_activation=nn.ReLU(),
    ):

        super(HGAT, self).__init__()

        self.gat_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        # input projection (no residual, no dropout)
        self.gat_layers.append(
            HGATConv(
                attn_mechanism,
                attn_order,
                in_feats,
                gat_hidden_size,
                num_heads,
                feat_drop=0,
                attn_drop=0,
                negative_slope=negative_slope,
                residual=False,
                unify_size=True,
            )
        )

        # hidden gat layers
        in_size = [gat_hidden_size for _ in in_feats]
        for _ in range(1, num_gat_layers):
            self.gat_layers.append(
                HGATConv(
                    attn_mechanism,
                    attn_order,
                    in_size,
                    gat_hidden_size,
                    num_heads,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    residual=True,
                    unify_size=False,
                )
            )

        # TODO this should be passed in as arguments
        canonical_etypes = [("atom", "a2b", "bond")]
        self.polling_layer = ConcatenateMeanMax(etypes=canonical_etypes)

        # hidden fc layer
        # 3 because we concatenate atom feats to bond feats in polling_layer
        in_size = gat_hidden_size * 3
        for _ in range(num_fc_layers - 1):
            self.fc_layers.append(nn.Linear(in_size, fc_hidden_size, bias=True))
            self.fc_layers.append(fc_activation)
            in_size = fc_hidden_size

        # output layer
        self.fc_layers.append(nn.Linear(in_size, 1, bias=True))

    def forward(self, graph, feats):
        h = feats

        # hgat layer
        for layer in self.gat_layers:
            h = layer(graph, h)

        # pooling layer
        h = self.polling_layer(graph, h)

        # fc
        # NOTE we add the 0 * h["atom"] + 0 * h["global"] to prevent GPU memory leak
        # this is actually should not happen, need carefully debug to figure out why
        h = h["bond"] + 0 * h["atom"].sum() + 0 * h["global"].sum()
        for layer in self.fc_layers:
            h = layer(h)
        h = h.view(-1)  # reshape to a 1D tensor to make each component a bond energy

        return h
