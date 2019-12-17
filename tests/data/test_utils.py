import numpy as np
from collections import defaultdict
from gnn.data.utils import StandardScaler
from ..utils import make_homo_graph, make_hetero_graph


def test_standard_scaler_hetero_graph():
    g1, feats1 = make_hetero_graph()
    g2, feats2 = make_hetero_graph()
    graphs = [g1, g2]

    ntypes = ["atom", "bond", "global"]
    ntypes = ["atom", "bond"]
    ref_feats = {nt: np.concatenate([feats1[nt], feats2[nt]]) for nt in ntypes}

    for k, v in ref_feats.items():
        mean = np.mean(v, axis=0)
        std = np.std(v, axis=0)
        v = (v - mean) / std
        ref_feats[k] = v

    # apply standardization
    scaler = StandardScaler()
    graphs = scaler(graphs)
    feats = defaultdict(list)
    for nt in ntypes:
        for g in graphs:
            feats[nt].append(g.nodes[nt].data["feat"])

    for nt in ntypes:
        ft = np.concatenate(feats[nt])
        assert np.allclose(ref_feats[nt], ft)


def test_standard_scaler_homo_graph():
    g1, feats1 = make_homo_graph()
    g2, feats2 = make_homo_graph()
    graphs = [g1, g2]

    ref_feats = {
        "node": np.concatenate([feats1["node"], feats2["node"]]),
        "edge": np.concatenate([feats1["edge"], feats2["edge"]]),
    }

    for k, v in ref_feats.items():
        mean = np.mean(v, axis=0)
        std = np.std(v, axis=0)
        v = (v - mean) / std
        ref_feats[k] = v

    # apply standardization
    scaler = StandardScaler()
    graphs = scaler(graphs)

    node_feats = []
    edge_feats = []
    for g in graphs:
        node_feats.append(g.ndata["feat"])
        edge_feats.append(g.edata["feat"])
    node_feats = np.concatenate(node_feats)
    edge_feats = np.concatenate(edge_feats)

    assert np.allclose(ref_feats["node"], node_feats)
    assert np.allclose(ref_feats["edge"], edge_feats)
