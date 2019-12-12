import torch
import numpy as np
from gnn.layer.utils import sum_nodes, softmax_nodes, broadcast_nodes
from ..utils import make_batched_hetero_graph


def test_sum_nodes():
    g, feats = make_batched_hetero_graph()
    rst = sum_nodes(g, "atom", "feat")

    num_nodes = 4
    size = 2
    ref = np.sum(np.arange(num_nodes * size).reshape(num_nodes, size), axis=0)
    assert np.allclose(rst, [ref, ref, ref])


def test_softmax_nodes():
    g, feats = make_batched_hetero_graph()
    rst = softmax_nodes(g, "atom", "feat")

    num_nodes = 4
    size = 2
    feat = np.arange(num_nodes * size).reshape(num_nodes, size)
    exp = np.exp(feat)
    ref = exp / np.sum(exp, axis=0)
    assert np.allclose(rst, np.concatenate([ref, ref, ref], axis=0))


def test_broadcast_nodes():
    g, feats = make_batched_hetero_graph()
    h = torch.tensor(
        np.arange(g.batch_size * 2).reshape(g.batch_size, 2), dtype=torch.float32
    )
    rst = broadcast_nodes(g, "atom", h)

    ref = np.repeat(h.numpy(), [4, 4, 4], axis=0)
    assert np.allclose(rst, ref)
