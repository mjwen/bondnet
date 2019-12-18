import numpy as np
import os
import torch
from collections import defaultdict
from gnn.data.electrolyte import ElectrolyteDataset
from gnn.data.qm9 import QM9Dataset
from gnn.data.dataloader import DataLoader, DataLoaderQM9
from gnn.data.utils import get_atom_to_bond_map, get_bond_to_atom_map


a_feat = torch.tensor(
    [
        [0.0, 1.0, 8.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
        [0.0, 1.0, 8.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 8.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
    ]
)
b_feat = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)
g_feat = torch.tensor([[0.0, 0.0, 1.0]])


def get_dataset():
    test_files = os.path.dirname(__file__)
    dataset = ElectrolyteDataset(
        sdf_file=os.path.join(test_files, "EC_struct.sdf"),
        label_file=os.path.join(test_files, "EC_label.txt"),
        self_loop=False,
    )
    for g, _ in dataset:
        g.nodes["atom"].data.update({"feat1": a_feat, "feat2": b_feat})
        g.nodes["bond"].data.update({"feat1": a_feat, "feat2": b_feat})
        g.nodes["global"].data.update({"feat_g": g_feat})
    return dataset


def get_qm9(properties, uc):
    test_files = os.path.dirname(__file__)
    dataset = QM9Dataset(
        sdf_file=os.path.join(test_files, "gdb9_n2.sdf"),
        label_file=os.path.join(test_files, "gdb9_n2.sdf.csv"),
        self_loop=False,
        properties=properties,
        unit_conversion=uc,
        normalize_extensive=False,
    )
    return dataset


def assert_graph_feature(g, num_graphs):
    if num_graphs == 1:
        assert np.allclose(g.nodes["atom"].data["feat1"], a_feat)
        assert np.allclose(g.nodes["atom"].data["feat2"], b_feat)
        assert np.allclose(g.nodes["bond"].data["feat1"], a_feat)
        assert np.allclose(g.nodes["bond"].data["feat2"], b_feat)
        assert np.allclose(g.nodes["global"].data["feat_g"], g_feat)
    elif num_graphs == 2:
        assert np.allclose(
            g.nodes["atom"].data["feat1"], np.concatenate((a_feat, a_feat))
        )
        assert np.allclose(
            g.nodes["atom"].data["feat2"], np.concatenate((b_feat, b_feat))
        )
        assert np.allclose(
            g.nodes["bond"].data["feat1"], np.concatenate((a_feat, a_feat))
        )
        assert np.allclose(
            g.nodes["bond"].data["feat2"], np.concatenate((b_feat, b_feat))
        )
        assert np.allclose(
            g.nodes["global"].data["feat_g"], np.concatenate((g_feat, g_feat))
        )
    else:
        raise ValueError("num_graphs not supported")


def assert_graph_struct(g, num_graphs):
    nodes = ["atom", "bond", "global"]
    edges = ["a2b", "b2a", "a2g", "g2a", "b2g", "g2b"]
    assert g.ntypes == nodes
    assert g.etypes == edges

    if num_graphs == 1:
        ref_num_nodes = [6, 6, 1]
        ref_num_edges = [12, 12, 6, 6, 6, 6]
        ref_b2a_map = {0: [0, 1], 1: [1, 3], 2: [2, 5], 3: [0, 2], 4: [1, 4], 5: [4, 5]}
    elif num_graphs == 2:
        ref_num_nodes = [12, 12, 2]
        ref_num_edges = [24, 24, 12, 12, 12, 12]
        ref_b2a_map = {0: [0, 1], 1: [1, 3], 2: [2, 5], 3: [0, 2], 4: [1, 4], 5: [4, 5]}
        ref_b2a_map2 = {k + 6: [i + 6 for i in v] for k, v in ref_b2a_map.items()}
        ref_b2a_map.update(ref_b2a_map2)
    else:
        raise ValueError("num_graphs not supported")

    ref_a2b_map = defaultdict(list)
    for b, atoms in ref_b2a_map.items():
        for a in atoms:
            ref_a2b_map[a].append(b)
    ref_a2b_map = {a: sorted(bonds) for a, bonds in ref_a2b_map.items()}

    num_nodes = [g.number_of_nodes(n) for n in nodes]
    num_edges = [g.number_of_edges(e) for e in edges]
    assert num_nodes == ref_num_nodes
    assert num_edges == ref_num_edges

    b2a_map = get_bond_to_atom_map(g)
    a2b_map = get_atom_to_bond_map(g)
    assert b2a_map == ref_b2a_map
    assert a2b_map == ref_a2b_map


def test_dataloader():
    # label references
    ref_label_energies = [[0, 0.1, 0, 0.2, 0, 0.3], [0.4, 0, 0, 0.5, 0, 0]]
    ref_label_indicators = [[0, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0]]

    dataset = get_dataset()

    # batch size 1 case
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (graph, labels) in enumerate(data_loader):

        # graph struct and feature
        assert_graph_struct(graph, num_graphs=1)
        assert_graph_feature(graph, num_graphs=1)

        # assert label
        assert np.allclose(labels["value"], ref_label_energies[i])
        assert np.allclose(labels["indicator"], ref_label_indicators[i])

    # batch size 2 case
    data_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    for graph, labels in data_loader:

        # graph struct and feature
        assert_graph_struct(graph, num_graphs=2)
        assert_graph_feature(graph, num_graphs=2)

        # assert label
        assert np.allclose(labels["value"], np.concatenate(ref_label_energies))
        assert np.allclose(labels["indicator"], np.concatenate(ref_label_indicators))


def test_dataloader_qm9():
    # label references
    properties = ["A", "h298_atom"]
    refs = {"A": [157.7118, 293.60975], "h298_atom": [-401.014646522, -280.399259105]}

    # batch size 1 case
    for prop in properties:
        dataset = get_qm9([prop], uc=False)
        data_loader = DataLoaderQM9(dataset, batch_size=1, shuffle=False)
        for i, (graph, labels) in enumerate(data_loader):
            r = np.atleast_2d(refs[prop][i])
            assert np.allclose(labels, r)

    # batch size 2 case
    for prop in properties:
        dataset = get_qm9([prop], uc=False)
        data_loader = DataLoaderQM9(dataset, batch_size=2, shuffle=False)
        for graph, labels in data_loader:
            r = np.asarray(refs[prop]).reshape(2, 1)
            assert np.allclose(labels, r)
