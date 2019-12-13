import numpy as np
from gnn.data.featurizer import (
    AtomFeaturizer,
    BondAsNodeFeaturizer,
    BondAsEdgeFeaturizer,
    MolChargeFeaturizer,
    MolWeightFeaturizer,
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
    featurizer = BondAsNodeFeaturizer()
    feat = featurizer(m)
    size = featurizer.feature_size
    assert np.array_equal(feat["feat"].shape, (m.GetNumBonds(), size))
    assert len(featurizer.feature_name) == size


def test_bond_as_edge_featurizer():
    def assert_featurizer(self_loop):
        m = make_EC_mol()
        featurizer = BondAsEdgeFeaturizer(self_loop=self_loop)
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
