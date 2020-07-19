import numpy as np
import os
from bondnet.data.electrolyte import (
    ElectrolyteBondDataset,
    ElectrolyteBondDatasetClassification,
    ElectrolyteMoleculeDataset,
    ElectrolyteReactionDataset,
    ElectrolyteReactionNetworkDataset,
)
from bondnet.data.qm9 import QM9Dataset
from bondnet.data.grapher import HeteroMoleculeGraph, HomoCompleteGraph
from bondnet.data.featurizer import (
    AtomFeaturizerFull,
    BondAsNodeFeaturizerFull,
    BondAsEdgeCompleteFeaturizer,
    GlobalFeaturizer,
)
import torch


test_files = os.path.join(os.path.dirname(__file__), "testdata")


def get_grapher_hetero():
    return HeteroMoleculeGraph(
        atom_featurizer=AtomFeaturizerFull(),
        bond_featurizer=BondAsNodeFeaturizerFull(),
        global_featurizer=GlobalFeaturizer(),
        self_loop=True,
    )


def get_grapher_homo():
    return HomoCompleteGraph(
        atom_featurizer=AtomFeaturizerFull(),
        bond_featurizer=BondAsEdgeCompleteFeaturizer(),
    )


def test_electrolyte_bond_label():
    def assert_label(lt):

        ref_label_energy = [[0.1, 0.2, 0.3], [0.4, 0.5]]
        ref_label_index = [[1, 3, 5], [0, 3]]

        if lt:
            energies = torch.tensor(np.concatenate(ref_label_energy))
            mean = float(torch.mean(energies))
            std = float(torch.std(energies))
            ref_label_energy = [(np.asarray(a) - mean) / std for a in ref_label_energy]
            ref_std = [[std] * len(x) for x in ref_label_energy]
            ref_mean = [[mean] * len(x) for x in ref_label_energy]

        dataset = ElectrolyteBondDataset(
            grapher=get_grapher_hetero(),
            molecules=os.path.join(test_files, "electrolyte_struct_bond.sdf"),
            labels=os.path.join(test_files, "electrolyte_label_bond.yaml"),
            extra_features=os.path.join(test_files, "electrolyte_feature_bond.yaml"),
            feature_transformer=True,
            label_transformer=lt,
        )

        size = len(dataset)
        assert size == 2

        for i in range(size):
            _, label = dataset[i]
            assert np.allclose(label["value"], ref_label_energy[i])
            assert np.array_equal(label["bond_index"], ref_label_index[i])
            if lt:
                assert np.allclose(label["scaler_mean"], ref_mean[i])
                assert np.allclose(label["scaler_stdev"], ref_std[i])
            else:
                assert "scaler_stedv" not in label

    assert_label(False)
    assert_label(True)


def test_electrolyte_bond_label_classification():

    ref_label_class = [0, 1]
    ref_label_indicators = [1, 2]

    dataset = ElectrolyteBondDatasetClassification(
        grapher=get_grapher_hetero(),
        molecules=os.path.join(test_files, "electrolyte_struct_bond.sdf"),
        labels=os.path.join(test_files, "electrolyte_label_bond_clfn.txt"),
        extra_features=os.path.join(test_files, "electrolyte_feature_bond.yaml"),
        feature_transformer=True,
    )

    size = len(dataset)
    assert size == 2

    for i in range(size):
        _, label = dataset[i]
        assert label["value"] == ref_label_class[i]
        assert label["indicator"] == ref_label_indicators[i]


def test_electrolyte_molecule_label():
    def assert_label(lt):
        ref_labels = np.asarray([[-0.941530613939904], [-8.91357537335352]])
        natoms = np.asarray([[2], [5]])

        if lt:
            ref_labels /= natoms
            ref_ts = natoms

        dataset = ElectrolyteMoleculeDataset(
            grapher=get_grapher_homo(),
            molecules=os.path.join(test_files, "electrolyte_struct_mol.sdf"),
            labels=os.path.join(test_files, "electrolyte_label_mol.csv"),
            properties=["atomization_energy"],
            unit_conversion=False,
            feature_transformer=True,
            label_transformer=lt,
        )

        size = len(dataset)
        assert size == 2

        for i in range(size):
            _, label = dataset[i]
            assert np.allclose(label["value"], ref_labels[i])
            if lt:
                assert np.allclose(label["scaler_stdev"], ref_ts[i])
            else:
                assert "scaler_stdev" not in label

    assert_label(False)
    assert_label(True)


def test_qm9_label():
    def assert_label(lt):
        ref_labels = np.asarray([[-0.3877, -40.47893], [-0.257, -56.525887]])
        natoms = [5, 4]

        if lt:
            homo = [ref_labels[0][0], ref_labels[1][0]]
            std = np.std(homo)
            homo = (homo - np.mean(homo)) / std
            for i in range(len(ref_labels)):
                ref_labels[i][0] = homo[i]
                ref_labels[i][1] /= natoms[i]
            ref_ts = [[std, natoms[0]], [std, natoms[1]]]

        dataset = QM9Dataset(
            grapher=get_grapher_homo(),
            molecules=os.path.join(test_files, "gdb9_n2.sdf"),
            labels=os.path.join(test_files, "gdb9_n2.sdf.csv"),
            properties=["homo", "u0"],  # homo is intensive and u0 is extensive
            unit_conversion=False,
            feature_transformer=True,
            label_transformer=lt,
        )

        size = len(dataset)
        assert size == 2

        for i in range(size):
            _, label = dataset[i]
            assert np.allclose(label["value"], ref_labels[i])
            if lt:
                assert np.allclose(label["scaler_stdev"], ref_ts[i])
            else:
                assert "scaler_stdev" not in label

    assert_label(False)
    assert_label(True)


def test_electrolyte_reaction_label():
    def assert_label(lt):
        ref_num_mols = [2, 3]
        ref_label_class = [0, 1]

        if lt:
            mean = np.mean(ref_label_class)
            std = np.std(ref_label_class)
            ref_label_class = (ref_label_class - mean) / std
            ref_ts = std

        dataset = ElectrolyteReactionDataset(
            grapher=get_grapher_hetero(),
            molecules=os.path.join(test_files, "electrolyte_struct_rxn_clfn.sdf"),
            labels=os.path.join(test_files, "electrolyte_label_rxn_clfn.yaml"),
            extra_features=os.path.join(test_files, "electrolyte_feature_rxn_clfn.yaml"),
            feature_transformer=True,
            label_transformer=lt,
        )

        size = len(dataset)
        assert size == 2

        for i in range(size):
            rxn, label = dataset[i]
            assert len(rxn) == label["num_mols"] == ref_num_mols[i]
            assert label["value"] == ref_label_class[i]

            if lt:
                assert label["scaler_stdev"] == ref_ts

    assert_label(False)
    assert_label(True)


def test_electrolyte_reaction_network_label():
    def assert_label(lt):
        ref_label_class = torch.tensor([0.0, 1.0])

        if lt:
            mean = torch.mean(ref_label_class)
            std = torch.std(ref_label_class)
            ref_label_class = (ref_label_class - mean) / std
            ref_ts = std

        dataset = ElectrolyteReactionNetworkDataset(
            grapher=get_grapher_hetero(),
            molecules=os.path.join(test_files, "electrolyte_struct_rxn_ntwk_clfn.sdf"),
            labels=os.path.join(test_files, "electrolyte_label_rxn_ntwk_clfn.yaml"),
            extra_features=os.path.join(
                test_files, "electrolyte_feature_rxn_ntwk_clfn.yaml"
            ),
            feature_transformer=True,
            label_transformer=lt,
        )

        size = len(dataset)
        assert size == 2

        for i in range(size):
            rn, rxn, label = dataset[i]
            assert label["value"] == ref_label_class[i]
            if lt:
                assert label["scaler_stdev"] == ref_ts

            assert rxn == i

            assert len(rn.molecules) == 5

    assert_label(False)
    assert_label(True)
