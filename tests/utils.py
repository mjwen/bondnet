import torch
import dgl
import numpy as np


def make_hetero_graph():
    r"""
    Make a heterograph for COHH and featurize it
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


def make_batched_hetero_graph(size=3):
    graphs = [make_hetero_graph()[0] for i in range(size)]
    g = dgl.batch_hetero(graphs)
    feats = {t: g.nodes[t].data["feat"] for t in ["atom", "bond", "global"]}
    return g, feats
