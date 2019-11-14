"""
Graph Attention Networks for heterograph. 
"""

import torch
import torch.nn as nn
import dgl.function as fn
from gnn.layer.hgatconv import HGATConv


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
        g,
        num_gat_layers=3,
        gat_hidden_size=64,
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

        master_nodes = (["atom", "bond", "global"],)
        attn_nodes = ([["bond", "global"], ["atom", "global"], ["atom", "bond"]],)
        attn_edges = ([["b2a", "g2a"], ["a2b", "g2b"], ["a2g", "b2g"]],)
        in_feats = [g.nodes[ntype].data["feat"].shape[1] for ntype in master_nodes]
        out_feats = gat_hidden_size

        self.gat_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        # input projection (no residual, no dropout)
        self.gat_layers.append(
            HGATConv(
                master_nodes,
                attn_nodes,
                attn_edges,
                in_feats,
                out_feats,
                num_heads,
                feat_drop=0,
                attn_drop=0,
                negative_slope=negative_slope,
                residual=False,
                unify_size=True,
            )
        )

        # hidden gat layers
        for _ in range(1, num_gat_layers):
            self.gat_layers.append(
                HGATConv(
                    master_nodes,
                    attn_nodes,
                    attn_edges,
                    in_feats,
                    out_feats,
                    num_heads,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    residual=True,
                    unify_size=False,
                )
            )

        # hidden fc layer
        h = out_feats
        for _ in range(num_fc_layers - 1):
            self.fc_layers.append(nn.Linear(h, fc_hidden_size, bias=True))
            self.fc_layers.append(fc_activation)
            h = fc_hidden_size

        # outout layer
        self.fc_layers.append(nn.Linear(h, 1, bias=True))

    def forward(self, inputs):
        h = inputs

        # hgat layer
        for layer in self.gat_layers:
            h = layer(self.g, h)

        # fc
        h = h["bond"]
        for layer in self.fc_layers:
            h = layer(h)
        return h
