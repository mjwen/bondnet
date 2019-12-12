import numpy as np
from gnn.data.featurizer import AtomFeaturizer, BondFeaturizer, GlobalStateFeaturizer
from .utils import make_EC_mol


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
    feat = featurizer(None, charge=1)
    assert featurizer.feature_size == 3
    size = featurizer.feature_size
    assert np.array_equal(feat["feat"].shape, (1, size))
    assert len(featurizer.feature_name) == size
