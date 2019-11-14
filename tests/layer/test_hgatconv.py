# pylint: disable=no-member

import torch
import dgl
import numpy as np
from gnn.layer.hgatconv import (
    UnifySize,
    NodeAttentionLayer,
    HGATConv,
    heterograph_edge_softmax,
)


def make_hetero_graph():
    r"""
    Make a heterograph for COHH
            O (0)
            || (0)
            C (1)
        /(1)  \ (2)
        H (2)  H (3)
    A global node u is attached to all atoms and bonds.
    """

    g = dgl.heterograph(
        {
            ("atom", "a2b", "bond"): [(0, 0), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)],
            ("bond", "b2a", "atom"): [(0, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 3)],
            ("atom", "a2g", "global"): [(i, 0) for i in range(4)],
            ("global", "g2a", "atom"): [(0, i) for i in range(4)],
            ("bond", "b2g", "global"): [(i, 0) for i in range(3)],
            ("global", "g2b", "bond"): [(0, i) for i in range(3)],
        }
    )
    return g


def test_unify_size():
    in_feats = {"a": 2, "b": 3}
    out_feats = 4
    us = UnifySize(in_feats, out_feats)

    feats = {"a": torch.zeros(2), "b": torch.zeros(3)}
    feats = us(feats)
    for k, v in feats.items():
        assert v.shape[0] == out_feats


def test_edge_softmax():

    g = make_hetero_graph()

    master_node = "atom"
    attn_nodes = ["bond", "global"]
    attn_edges = ["b2a", "g2a"]

    feat_size = 3
    edge_data = []
    for n, e in zip(attn_nodes, attn_edges):
        etype = (n, e, master_node)
        nedge = g.number_of_edges(etype)
        d = np.asarray(
            np.arange(nedge * feat_size).reshape(nedge, feat_size), dtype=np.float32
        )
        edge_data.append(torch.from_numpy(d))
    a = heterograph_edge_softmax(g, master_node, attn_nodes, attn_edges, edge_data)

    # check the softmax for edges connected to atom 1
    # atom 1 is connected to bond 0,1,2 (with bond edge indices 1,2,4) and the single
    # global node (global edge index 1) it is data is
    x = [[3, 4, 5], [6, 7, 8], [12, 13, 14], [3, 4, 5]]
    # softmax
    x = np.exp(x)
    x_sum = np.sum(x, axis=0)
    x = x / x_sum

    # bond edges, [0]
    assert np.allclose(a[0][[1, 2, 4]], x[:3])
    # global edges, [1]
    assert np.allclose(a[1][1], x[3])


def test_node_attn_layer():
    in_feats = 3
    out_feats = 8
    num_heads = 2
    attn_layer = NodeAttentionLayer(
        master_node="atom",
        attn_nodes=["bond", "global"],
        attn_edges=["b2a", "g2a"],
        in_feats=in_feats,
        out_feats=out_feats,
        num_heads=num_heads,
    )

    g = make_hetero_graph()
    natoms = g.number_of_nodes("atom")
    nbonds = g.number_of_nodes("bond")
    atom_feats = torch.randn(natoms, in_feats)
    bond_feats = torch.randn(nbonds, in_feats)
    global_feats = torch.randn(1, in_feats)

    out = attn_layer(g, master_feats=atom_feats, attn_feats=[bond_feats, global_feats])

    assert np.array_equal(out.shape, [natoms, num_heads, out_feats])


def test_hgat_conv_layer():

    in_feats = [2, 3, 4]
    out_feats = 5
    num_heads = 2
    master_nodes = ["atom", "bond", "global"]
    attn_nodes = [["bond", "global"], ["atom", "global"], ["atom", "bond"]]
    attn_edges = [["b2a", "g2a"], ["a2b", "g2b"], ["a2g", "b2g"]]
    gat_layer = HGATConv(
        master_nodes,
        attn_nodes,
        attn_edges,
        in_feats,
        out_feats,
        num_heads,
        unify_size=True,
    )

    g = make_hetero_graph()
    num_nodes = [g.number_of_nodes("atom"), g.number_of_nodes("bond"), 1]

    feats = {}
    ntype2num = {}
    for ntype, ift, num in zip(master_nodes, in_feats, num_nodes):
        feats[ntype] = torch.randn(num, ift)
        ntype2num[ntype] = num
    out = gat_layer(g, feats)

    assert set(out.keys()) == set(master_nodes)
    for k, v in out.items():
        assert np.array_equal(v.shape, [ntype2num[k], out_feats])

