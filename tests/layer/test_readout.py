# pylint: disable=no-member
import torch
import dgl
import numpy as np
from gnn.layer.readout import ConcatenateMeanMax, Set2Set, Set2SetThenCat


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
            ("atom", "a2b", "bond"): [(0, 0), (1, 0), (1, 1), (1, 2), (2, 1), (3, 2)],
            ("bond", "b2a", "atom"): [(0, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 3)],
            ("atom", "a2g", "global"): [(i, 0) for i in range(4)],
            ("global", "g2a", "atom"): [(0, i) for i in range(4)],
            ("bond", "b2g", "global"): [(i, 0) for i in range(3)],
            ("global", "g2b", "bond"): [(0, i) for i in range(3)],
        }
    )
    feats_size = {"atom": 2, "bond": 3, "global": 4}

    feats = {}
    for ntype, size in feats_size.items():
        num_node = g.number_of_nodes(ntype)
        ft = torch.tensor(
            np.arange(num_node * size).reshape(num_node, size), dtype=torch.float32
        )
        g.nodes[ntype].data.update({"feat": ft})
        feats[ntype] = ft

    return g, feats


def make_batch(size=3):
    graphs = [make_hetero_graph()[0] for i in range(size)]
    g = dgl.batch_hetero(graphs)
    feats = {t: g.nodes[t].data["feat"] for t in ["atom", "bond", "global"]}
    return g, feats


def test_concatenate_mean_max():
    g, feats = make_hetero_graph()

    etypes = [("bond", "b2a", "atom")]
    layer = ConcatenateMeanMax(etypes)
    rst = layer(g, feats)

    ref = [
        [0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        [2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [4.0, 5.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 6.0, 7.0, 8.0, 6.0, 7.0, 8.0],
    ]

    assert np.allclose(rst["atom"], ref)


def test_set2set():
    nbatch = 3
    g, feats = make_batch(nbatch)

    in_feat = 2
    ntype = "atom"
    layer = Set2Set(input_dim=in_feat, n_iters=2, n_layers=4, ntype=ntype)
    rst = layer(g, feats[ntype])
    assert rst.shape == (nbatch, in_feat * 2)


def test_set2set_then_cat():
    nbatch = 3
    g, feats = make_batch(nbatch)

    layer = Set2SetThenCat(
        n_iters=2,
        n_layer=4,
        ntypes=["atom", "bond"],
        in_feats=[2, 3, 4],
        ntypes_direct_cat=["global"],
    )
    rst = layer(g, feats)
    assert rst.shape == (nbatch, 2 * 2 + 3 * 2 + 4)
