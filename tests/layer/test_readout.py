import numpy as np
from gnn.layer.readout import ConcatenateMeanMax, Set2Set, Set2SetThenCat
from ..utils import make_hetero_graph, make_batched_hetero_graph


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
    g, feats = make_batched_hetero_graph(nbatch)

    in_feat = 2
    ntype = "atom"
    layer = Set2Set(input_dim=in_feat, n_iters=2, n_layers=4, ntype=ntype)
    rst = layer(g, feats[ntype])
    assert rst.shape == (nbatch, in_feat * 2)


def test_set2set_then_cat():
    nbatch = 3
    g, feats = make_batched_hetero_graph(nbatch)

    layer = Set2SetThenCat(
        n_iters=2,
        n_layer=4,
        ntypes=["atom", "bond"],
        in_feats=[2, 3, 4],
        ntypes_direct_cat=["global"],
    )
    rst = layer(g, feats)
    assert rst.shape == (nbatch, 2 * 2 + 3 * 2 + 4)
