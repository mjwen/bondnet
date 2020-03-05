"""
Do not assert feature and graph struct, which is handled by dgl.
Here we mainly test the correctness of batch.
"""

import numpy as np
import os
from gnn.data.electrolyte import (
    ElectrolyteBondDataset,
    ElectrolyteBondDatasetClassification,
    ElectrolyteReactionDataset,
)
from gnn.data.qm9 import QM9Dataset
from gnn.data.dataloader import (
    DataLoaderBond,
    DataLoaderMolecule,
    DataLoaderBondClassification,
    DataLoaderReaction,
)
from gnn.data.grapher import HeteroMoleculeGraph, HomoCompleteGraph
from gnn.data.featurizer import (
    AtomFeaturizer,
    BondAsNodeFeaturizer,
    BondAsEdgeCompleteFeaturizer,
    GlobalFeaturizerChargeSpin,
)


test_files = os.path.dirname(__file__)


def get_grapher_hetero():
    return HeteroMoleculeGraph(
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondAsNodeFeaturizer(),
        global_featurizer=GlobalFeaturizerChargeSpin(),
        self_loop=True,
    )


def get_grapher_homo():
    return HomoCompleteGraph(
        atom_featurizer=AtomFeaturizer(), bond_featurizer=BondAsEdgeCompleteFeaturizer()
    )


def test_dataloader_bond():
    def assert_label(lt):
        ref_label_energies = [[0, 0.1, 0, 0.2, 0, 0.3], [0.4, 0, 0, 0.5, 0, 0]]
        ref_label_indicators = [[0, 1, 0, 1, 0, 1], [1, 0, 0, 1, 0, 0]]

        if lt:
            non_zeros = [i for j in ref_label_energies for i in j if i != 0.0]
            mean = np.mean(non_zeros)
            std = np.std(non_zeros)
            ref_label_energies = [
                (np.asarray(a) - mean) / std for a in ref_label_energies
            ]
            ref_scales = [[std] * len(x) for x in ref_label_energies]

        dataset = ElectrolyteBondDataset(
            grapher=get_grapher_hetero(),
            sdf_file=os.path.join(test_files, "electrolyte_struct_bond.sdf"),
            label_file=os.path.join(test_files, "electrolyte_label_bond.txt"),
            feature_file=os.path.join(test_files, "electrolyte_feature_bond.yaml"),
            feature_transformer=False,
            label_transformer=lt,
        )

        # batch size 1 case (exactly the same as test_dataset)
        data_loader = DataLoaderBond(dataset, batch_size=1, shuffle=False)
        for i, (graph, labels, scales) in enumerate(data_loader):
            assert np.allclose(labels["value"], ref_label_energies[i])
            assert np.allclose(labels["indicator"], ref_label_indicators[i])
            if lt:
                assert np.allclose(scales, ref_scales[i])
            else:
                assert scales is None

        # batch size 2 case
        data_loader = DataLoaderBond(dataset, batch_size=2, shuffle=False)
        for graph, labels, scales in data_loader:
            assert np.allclose(labels["value"], np.concatenate(ref_label_energies))
            assert np.allclose(labels["indicator"], np.concatenate(ref_label_indicators))
            if lt:
                assert np.allclose(scales, np.concatenate(ref_scales))
            else:
                assert scales is None

    assert_label(False)
    assert_label(True)


def test_dataloader_molecule():
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
            ref_scales = [[std, natoms[0]], [std, natoms[1]]]

        dataset = QM9Dataset(
            grapher=get_grapher_homo(),
            sdf_file=os.path.join(test_files, "gdb9_n2.sdf"),
            label_file=os.path.join(test_files, "gdb9_n2.sdf.csv"),
            properties=["homo", "u0"],  # homo is intensive and u0 is extensive
            unit_conversion=False,
            feature_transformer=True,
            label_transformer=lt,
        )

        # batch size 1 case (exactly the same as test_dataset)
        data_loader = DataLoaderMolecule(dataset, batch_size=1, shuffle=False)
        for i, (graph, labels, scales) in enumerate(data_loader):
            assert np.allclose(labels, [ref_labels[i]])
            if lt:
                assert np.allclose(scales, [ref_scales[i]])
            else:
                assert scales is None

        # batch size 2 case
        data_loader = DataLoaderMolecule(dataset, batch_size=2, shuffle=False)
        for graph, labels, scales in data_loader:
            assert np.allclose(labels, ref_labels)
            if lt:
                assert np.allclose(scales, ref_scales)
            else:
                assert scales is None


def test_dataloader_bond_classification():
    ref_label_class = [0, 1]
    ref_label_indicators = [1, 2]

    dataset = ElectrolyteBondDatasetClassification(
        grapher=get_grapher_hetero(),
        sdf_file=os.path.join(test_files, "electrolyte_struct_bond.sdf"),
        label_file=os.path.join(test_files, "electrolyte_label_bond_clfn.txt"),
        feature_file=os.path.join(test_files, "electrolyte_feature_bond.yaml"),
        feature_transformer=False,
    )

    # batch size 1 case (exactly the same as test_dataset)
    data_loader = DataLoaderBondClassification(dataset, batch_size=1, shuffle=False)
    for i, (graph, labels) in enumerate(data_loader):
        assert np.allclose(labels["class"], ref_label_class[i])
        assert np.allclose(labels["indicator"], ref_label_indicators[i])

    # batch size 2 case
    data_loader = DataLoaderBondClassification(dataset, batch_size=2, shuffle=False)
    for graph, labels in data_loader:
        assert np.allclose(labels["class"], ref_label_class)
        assert np.allclose(labels["indicator"], ref_label_indicators)


def test_dataloader_reaction():
    ref_label_class = [0, 0]
    ref_num_mols = [2, 3]

    dataset = ElectrolyteReactionDataset(
        grapher=get_grapher_hetero(),
        sdf_file=os.path.join(test_files, "electrolyte_struct_rxn_clfn.sdf"),
        label_file=os.path.join(test_files, "electrolyte_label_rxn_clfn.yaml"),
        feature_file=os.path.join(test_files, "electrolyte_feature_rxn_clfn.yaml"),
        feature_transformer=False,
    )

    # batch size 1 case (exactly the same as test_dataset)
    data_loader = DataLoaderReaction(dataset, batch_size=1, shuffle=False)
    for i, (graph, labels) in enumerate(data_loader):
        assert np.allclose(labels["value"], ref_label_class[i])
        assert np.allclose(labels["num_mols"], ref_num_mols[i])

    # batch size 2 case
    data_loader = DataLoaderReaction(dataset, batch_size=2, shuffle=False)
    for graph, labels in data_loader:
        assert np.allclose(labels["value"], ref_label_class)
        assert np.allclose(labels["num_mols"], ref_num_mols)
