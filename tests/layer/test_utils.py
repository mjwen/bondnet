# pylint: disable=no-member
import torch
import dgl
import numpy as np
from gnn.layer.utils import sum_nodes, softmax_nodes, broadcast_nodes


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


def test_sum_nodes():
    g, feats = make_batch()
    rst = sum_nodes(g, "atom", "feat")

    num_nodes = 4
    size = 2
    ref = np.sum(np.arange(num_nodes * size).reshape(num_nodes, size), axis=0)
    assert np.allclose(rst, [ref, ref, ref])


def test_softmax_nodes():
    g, feats = make_batch()
    rst = softmax_nodes(g, "atom", "feat")

    num_nodes = 4
    size = 2
    feat = np.arange(num_nodes * size).reshape(num_nodes, size)
    exp = np.exp(feat)
    ref = exp / np.sum(exp, axis=0)
    assert np.allclose(rst, np.concatenate([ref, ref, ref], axis=0))


def test_broadcast_nodes():
    g, feats = make_batch()
    h = torch.tensor(
        np.arange(g.batch_size * 2).reshape(g.batch_size, 2), dtype=torch.float32
    )
    rst = broadcast_nodes(g, "atom", h)

    ref = np.repeat(h.numpy(), [4, 4, 4], axis=0)
    assert np.allclose(rst, ref)
