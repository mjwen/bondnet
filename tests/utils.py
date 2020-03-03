import torch
import dgl
import numpy as np


def make_hetero(num_atoms, num_bonds, a2b, b2a):
    """
    Create a hetero graph and create features.
    A global node is connected to all atoms and bonds.

    Atom features are:
    [[0,1],
     [2,3],
     .....]

    Bond features are:
    [[0,1,2],
     [3,4,5],
     .....]

    Global features are:
    [[0,1,2,3]]
    """
    if num_bonds == 0:
        # create a fake bond and create an edge atom->bond
        num_bonds = 1
        a2b = [(0, 0)]
        b2a = [(0, 0)]

    g = dgl.heterograph(
        {
            ("atom", "a2b", "bond"): a2b,
            ("bond", "b2a", "atom"): b2a,
            ("atom", "a2g", "global"): [(i, 0) for i in range(num_atoms)],
            ("global", "g2a", "atom"): [(0, i) for i in range(num_atoms)],
            ("bond", "b2g", "global"): [(i, 0) for i in range(num_bonds)],
            ("global", "g2b", "bond"): [(0, i) for i in range(num_bonds)],
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


def make_hetero_CH2O():
    r"""
            O (0)
            || (0)
            C (1)
        /(1)  \ (2)
        H (2)  H (3)

    atom features:
    [[0,1],
     [2,3],
     [4,5],
     [6,7],]

    bond features:
    [[0,1,2],
     [3,4,5],
     [6,7,8]]

    global features:
    [[0,1,2,3]]
    """

    return make_hetero(
        num_atoms=4,
        num_bonds=3,
        a2b=[(0, 0), (1, 0), (1, 1), (1, 2), (2, 1), (3, 2)],
        b2a=[(0, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 3)],
    )


def make_hetero_CHO():
    r"""
            O (0)
            || (0)
            C (1)
        /(1)
        H (2)

    atom features:
    [[0,1],
     [2,3],
     [4,5],]

    bond features:
    [[0,1,2],
     [3,4,5]]

    global features:
    [[0,1,2,3]]
    """

    return make_hetero(
        num_atoms=3,
        num_bonds=2,
        a2b=[(0, 0), (1, 0), (1, 1), (2, 1)],
        b2a=[(0, 0), (0, 1), (1, 1), (1, 2)],
    )


def make_hetero_H():
    r"""
    H

    atom features:
    [[0,1]]

    bond features:
    []

    global features:
    [[0,1,2,3]]
    """

    return make_hetero(num_atoms=1, num_bonds=0, a2b=[], b2a=[])


def make_batched_hetero_CH2O(size=3):
    graphs = [make_hetero_CH2O()[0] for i in range(size)]
    g = dgl.batch_hetero(graphs)
    feats = {t: g.nodes[t].data["feat"] for t in ["atom", "bond", "global"]}
    return g, feats


def make_batched_hetero_forming_reaction():
    graphs = [
        make_hetero_CH2O()[0],
        make_hetero_CHO()[0],
        make_hetero_H()[0],
        make_hetero_CH2O()[0],
        make_hetero_CHO()[0],
        make_hetero_H()[0],
    ]
    g = dgl.batch_hetero(graphs)
    feats = {t: g.nodes[t].data["feat"] for t in ["atom", "bond", "global"]}
    return g, feats


def make_homo_CH2O():
    r"""
    Make a bidirected homograph for COHH and featurize it
            O (0)
            || (0)
            C (1)
        /(1)  \ (2)
        H (2)  H (3)
    A global node u is attached to all atoms and bonds.
    """

    g = dgl.DGLGraph()
    g.add_nodes(4)
    src = [0, 1, 1, 1, 2, 3]
    des = [1, 0, 2, 3, 1, 1]
    g.add_edges(src, des)

    feats = {}
    N = 4
    size = 2
    ft = torch.tensor(np.arange(N * size).reshape(N, size), dtype=torch.float32)
    g.ndata.update({"feat": ft})
    feats["node"] = ft

    N = 6
    size = 3
    ft = torch.tensor(np.arange(N * size).reshape(N, size), dtype=torch.float32)
    g.edata.update({"feat": ft})
    feats["edge"] = ft

    return g, feats
