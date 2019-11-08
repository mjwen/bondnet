import numpy as np
from collections import defaultdict
import torch
from rdkit import Chem
from gnn.data.featurizer import (
    AtomFeaturizer,
    BondFeaturizer,
    GlobalStateFeaturizer,
    HeteroMoleculeGraph,
)

a_feat = [
    [0.0, 1.0, 8.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
    [0.0, 1.0, 8.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 8.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
]
b_feat = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]
g_feat = [0.0, 0.0, 1.0]


def make_EC_mol():
    sdf = """EC  C1COC(=O)O1
 OpenBabel10281912183D

  6  6  0  0  0  0  0  0  0  0999 V2000
    0.2938   -1.2494    0.0142 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7512   -0.4581    0.0690 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5279   -0.5231   -0.0831 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.8980   -0.6830    0.1536 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.2610    0.9193    0.0069 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.0727    0.9009   -0.0656 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  4  2  0  0  0  0
  3  6  1  0  0  0  0
  3  1  1  0  0  0  0
  5  2  1  0  0  0  0
  6  5  1  0  0  0  0
M  END
$$$$
    """
    return Chem.MolFromMolBlock(sdf)


def test_atom_featurizer():
    node_type = np.array([8, 6, 6, 8, 8, 6])

    m = make_EC_mol()
    species = list(set([a.GetSymbol() for a in m.GetAtoms()]))
    featurizer = AtomFeaturizer(species)
    feat = featurizer(m)
    assert featurizer.feature_size == 10
    assert np.array_equal(feat["node_type"], node_type)
    assert np.allclose(feat["a_feat"], a_feat)


def test_bond_featurizer():
    m = make_EC_mol()
    featurizer = BondFeaturizer()
    feat = featurizer(m)
    assert featurizer.feature_size == 6
    assert np.allclose(feat["b_feat"], b_feat)


def test_global_state_featurizer():
    featurizer = GlobalStateFeaturizer()
    feat = featurizer(charge=1)
    assert featurizer.feature_size == 3
    assert np.allclose(feat["g_feat"], g_feat)


def test_build_graph():
    m = make_EC_mol()
    grapher = HeteroMoleculeGraph()
    g = grapher.build_graph(m)

    nodes = ["atom", "bond", "global"]
    edges = ["anb", "bna", "ang", "gna", "bng", "gnb"]
    assert g.ntypes == nodes
    assert g.etypes == edges
    num_nodes = [g.number_of_nodes(n) for n in nodes]
    assert num_nodes == [6, 6, 1]
    num_edges = [g.number_of_edges(e) for e in edges]
    assert num_edges == [12, 12, 6, 6, 6, 6]

    # this is in info contained in the sdf file (note dgl graph index starts from 0)
    bond_to_atom_map = {0: [0, 1], 1: [1, 3], 2: [2, 5], 3: [0, 2], 4: [1, 4], 5: [4, 5]}
    atom_to_bond_map = defaultdict(list)
    for b, atoms in bond_to_atom_map.items():
        for a in atoms:
            atom_to_bond_map[a].append(b)
    atom_to_bond_map = {a: sorted(bonds) for a, bonds in atom_to_bond_map.items()}

    ref_b2a_map = grapher.get_bond_to_atom_map(g)
    ref_a2b_map = grapher.get_atom_to_bond_map(g)
    assert bond_to_atom_map == ref_b2a_map
    assert atom_to_bond_map == ref_a2b_map


def test_graph_featurize():
    # make sure the others work and then this should work
    m = make_EC_mol()
    species = list(set([a.GetSymbol() for a in m.GetAtoms()]))
    charge = 1
    grapher = HeteroMoleculeGraph(
        atom_featurizer=AtomFeaturizer(species),
        bond_featurizer=BondFeaturizer(),
        global_state_featurizer=GlobalStateFeaturizer(),
    )
    g = grapher.build_graph_and_featurize(m, charge)
    assert np.allclose(g.nodes["atom"].data["a_feat"], a_feat)
    assert np.allclose(g.nodes["bond"].data["b_feat"], b_feat)
    assert np.allclose(g.nodes["global"].data["g_feat"], g_feat)
