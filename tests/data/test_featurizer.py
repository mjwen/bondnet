import numpy as np
from collections import defaultdict
from rdkit import Chem
from gnn.data.featurizer import (
    AtomFeaturizer,
    BondFeaturizer,
    GlobalStateFeaturizer,
    HeteroMoleculeGraph,
)
from gnn.data.utils import get_atom_to_bond_map, get_bond_to_atom_map


def make_EC_mol():
    sdf = """5d1a79e59ab9e0c05b1de572
 OpenBabel11151914373D

  7  7  0  0  0  0  0  0  0  0999 V2000
    0.0852   -0.2958   -0.5026 O   0  3  0  0  0  0  0  0  0  0  0  0
    1.4391   -0.0921   -0.0140 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.1032   -1.4653    0.0152 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.9476   -2.1383   -1.1784 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.3604   -1.7027   -1.7840 Li  0  5  0  0  0  0  0  0  0  0  0  0
   -0.3721    0.5500   -0.5368 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.4039    0.3690    0.9792 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  5  1  0  0  0  0
  2  3  1  0  0  0  0
  2  7  1  0  0  0  0
  4  3  1  0  0  0  0
  5  4  1  0  0  0  0
  6  1  1  0  0  0  0
M  CHG  2   1   1   5  -1
M  ZCH  2   1   0   5   0
M  ZBO  1   2   0
M  END
$$$$
    """
    return Chem.MolFromMolBlock(sdf, sanitize=True, removeHs=False)


def test_atom_featurizer():
    m = make_EC_mol()
    species = list(set([a.GetSymbol() for a in m.GetAtoms()]))
    featurizer = AtomFeaturizer(species)
    feat = featurizer(m)
    size = featurizer.feature_size
    assert np.array_equal(feat["feat"].shape, (m.GetNumAtoms(), size))
    assert len(featurizer.feature_name) == size


def test_bond_featurizer():
    m = make_EC_mol()
    featurizer = BondFeaturizer()
    feat = featurizer(m)
    size = featurizer.feature_size
    assert np.array_equal(feat["feat"].shape, (m.GetNumAtoms(), size))
    assert len(featurizer.feature_name) == size


def test_global_state_featurizer():
    featurizer = GlobalStateFeaturizer()
    feat = featurizer(charge=1)
    assert featurizer.feature_size == 3
    size = featurizer.feature_size
    assert np.array_equal(feat["feat"].shape, (1, size))
    assert len(featurizer.feature_name) == size


def test_build_graph():
    m = make_EC_mol()
    grapher = HeteroMoleculeGraph()
    g = grapher.build_graph(m)

    nodes = ["atom", "bond", "global"]
    edges = ["a2b", "b2a", "a2g", "g2a", "b2g", "g2b"]
    assert g.ntypes == nodes
    assert g.etypes == edges
    num_nodes = [g.number_of_nodes(n) for n in nodes]
    assert num_nodes == [7, 7, 1]
    num_edges = [g.number_of_edges(e) for e in edges]
    assert num_edges == [14, 14, 7, 7, 7, 7]

    bond_to_atom_map = {
        0: [0, 1],
        1: [0, 4],
        2: [1, 2],
        3: [1, 6],
        4: [2, 3],
        5: [3, 4],
        6: [0, 5],
    }
    atom_to_bond_map = defaultdict(list)
    for b, atoms in bond_to_atom_map.items():
        for a in atoms:
            atom_to_bond_map[a].append(b)
    atom_to_bond_map = {a: sorted(bonds) for a, bonds in atom_to_bond_map.items()}

    ref_b2a_map = get_bond_to_atom_map(g)
    ref_a2b_map = get_atom_to_bond_map(g)
    assert bond_to_atom_map == ref_b2a_map
    assert atom_to_bond_map == ref_a2b_map


def test_graph_featurize():
    # make sure the others work and then this should work
    m = make_EC_mol()
    species = list(set([a.GetSymbol() for a in m.GetAtoms()]))
    charge = 1

    atom_featurizer = AtomFeaturizer(species)
    bond_featurizer = BondFeaturizer()
    global_state_featurizer = GlobalStateFeaturizer()
    grapher = HeteroMoleculeGraph(
        atom_featurizer, bond_featurizer, global_state_featurizer
    )
    g = grapher.build_graph_and_featurize(m, charge)
    assert np.array_equal(
        g.nodes["atom"].data["feat"].shape,
        (m.GetNumAtoms(), atom_featurizer.feature_size),
    )
    assert np.array_equal(
        g.nodes["bond"].data["feat"].shape,
        (m.GetNumBonds(), bond_featurizer.feature_size),
    )
    assert np.array_equal(
        g.nodes["global"].data["feat"].shape, (1, global_state_featurizer.feature_size)
    )
