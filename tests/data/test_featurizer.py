import numpy as np
from gnn.data.featurizer import (
    AtomFeaturizer,
    BondAsNodeFeaturizer,
    BondAsEdgeBidirectedFeaturizer,
    BondAsEdgeCompleteFeaturizer,
    MolChargeFeaturizer,
    MolWeightFeaturizer,
    DistanceBins,
)
from .utils import make_EC_mol


def test_atom_featurizer():
    m = make_EC_mol()
    species = list(set([a.GetSymbol() for a in m.GetAtoms()]))
    featurizer = AtomFeaturizer(species)
    feat = featurizer(m)
    size = featurizer.feature_size
    assert np.array_equal(feat["feat"].shape, (m.GetNumAtoms(), size))
    assert len(featurizer.feature_name) == size


def test_bond_as_node_featurizer():
    m = make_EC_mol()
    featurizer = BondAsNodeFeaturizer(length_featurizer="bin")
    feat = featurizer(m)
    size = featurizer.feature_size
    assert np.array_equal(feat["feat"].shape, (m.GetNumBonds(), size))
    assert len(featurizer.feature_name) == size


def test_bond_as_edge_bidirected_featurizer():
    def assert_featurizer(self_loop):
        m = make_EC_mol()
        featurizer = BondAsEdgeBidirectedFeaturizer(
            self_loop=self_loop, length_featurizer="bin"
        )
        feat = featurizer(m)
        size = featurizer.feature_size

        natoms = m.GetNumAtoms()
        nbonds = m.GetNumBonds()
        if self_loop:
            nedges = 2 * nbonds + natoms
        else:
            nedges = 2 * nbonds

        assert np.array_equal(feat["feat"].shape, (nedges, size))
        assert len(featurizer.feature_name) == size

    assert_featurizer(True)
    assert_featurizer(False)


def test_bond_as_edge_complete_featurizer():
    def assert_featurizer(self_loop):
        m = make_EC_mol()
        featurizer = BondAsEdgeCompleteFeaturizer(
            self_loop=self_loop, length_featurizer="bin"
        )
        feat = featurizer(m)
        size = featurizer.feature_size

        natoms = m.GetNumAtoms()
        if self_loop:
            nedges = natoms ** 2
        else:
            nedges = natoms * (natoms - 1)

        assert np.array_equal(feat["feat"].shape, (nedges, size))
        assert len(featurizer.feature_name) == size

    assert_featurizer(True)
    assert_featurizer(False)


def test_mol_charge_featurizer():
    featurizer = MolChargeFeaturizer()
    feat = featurizer(None, charge=1)
    size = featurizer.feature_size
    assert size == 3
    assert np.array_equal(feat["feat"].shape, (1, size))
    assert len(featurizer.feature_name) == size


def test_mol_weight_featurizer():
    m = make_EC_mol()
    featurizer = MolWeightFeaturizer()
    feat = featurizer(m)
    size = featurizer.feature_size
    assert size == 3
    assert np.array_equal(feat["feat"].shape, (1, size))
    assert len(featurizer.feature_name) == size


def test_dist_bins():
    dist_b = DistanceBins(low=2, high=6, num_bins=10)
    print(dist_b.bins)

    ref = np.zeros(10)
    ref[1] = 1
    assert np.array_equal(dist_b(2), ref)

    ref = np.zeros(10)
    ref[0] = 1
    assert np.array_equal(dist_b(1.9999), ref)

    ref = np.zeros(10)
    ref[9] = 1
    assert np.array_equal(dist_b(6), ref)

    ref = np.zeros(10)
    ref[8] = 1
    assert np.array_equal(dist_b(5.9999), ref)
