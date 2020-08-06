"""
Do not assert feature and graph struct, which is handled by dgl.
Here we mainly test the correctness of batch.
"""

from pathlib import Path
import torch
import numpy as np
from bondnet.data.electrolyte import (
    ElectrolyteBondDataset,
    ElectrolyteReactionDataset,
    ElectrolyteReactionNetworkDataset,
)
from bondnet.data.qm9 import QM9Dataset
from bondnet.data.dataloader import (
    DataLoaderBond,
    DataLoader,
    DataLoaderReaction,
    DataLoaderReactionNetwork,
)
from bondnet.data.grapher import HeteroMoleculeGraph, HomoCompleteGraph
from bondnet.data.featurizer import (
    AtomFeaturizerFull,
    BondAsNodeFeaturizerFull,
    BondAsEdgeCompleteFeaturizer,
    GlobalFeaturizer,
)

test_files = Path(__file__).parent.joinpath("testdata")


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


def test_dataloader():
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
            molecules=test_files.joinpath("gdb9_n2.sdf"),
            labels=test_files.joinpath("gdb9_n2.sdf.csv"),
            properties=["homo", "u0"],  # homo is intensive and u0 is extensive
            unit_conversion=False,
            feature_transformer=True,
            label_transformer=lt,
        )

        # batch size 1 case (exactly the same as test_dataset)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        for i, (graph, labels) in enumerate(data_loader):
            assert np.allclose(labels["value"], [ref_labels[i]])
            if lt:
                assert np.allclose(labels["scaler_stdev"], [ref_scales[i]])

        # batch size 2 case
        data_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        for graph, labels in data_loader:
            assert np.allclose(labels["value"], ref_labels)
            if lt:
                assert np.allclose(labels["scaler_stdev"], ref_scales)

    assert_label(False)
    assert_label(True)


def test_dataloader_bond():
    def assert_label(lt):

        ref_label_energy = [[0.1, 0.2, 0.3], [0.4, 0.5]]
        ref_label_index = [[1, 3, 5], [0, 3]]
        ref_label_index_2 = [1, 3, 5, 6, 9]

        if lt:
            energies = torch.tensor(np.concatenate(ref_label_energy))
            mean = float(torch.mean(energies))
            std = float(torch.std(energies))
            ref_label_energy = [(np.asarray(a) - mean) / std for a in ref_label_energy]
            ref_mean = [[mean] * len(x) for x in ref_label_energy]
            ref_std = [[std] * len(x) for x in ref_label_energy]

        dataset = ElectrolyteBondDataset(
            grapher=get_grapher_hetero(),
            molecules=test_files.joinpath("electrolyte_struct_bond.sdf"),
            labels=test_files.joinpath("electrolyte_label_bond.yaml"),
            extra_features=test_files.joinpath("electrolyte_feature_bond.yaml"),
            feature_transformer=True,
            label_transformer=lt,
        )

        # batch size 1 case (exactly the same as test_dataset)
        data_loader = DataLoaderBond(dataset, batch_size=1, shuffle=False)
        for i, (graph, labels) in enumerate(data_loader):
            assert np.allclose(labels["value"], ref_label_energy[i])
            assert np.allclose(labels["index"], ref_label_index[i])
            if lt:
                assert np.allclose(labels["scaler_mean"], ref_mean[i])
                assert np.allclose(labels["scaler_stdev"], ref_std[i])

        # batch size 2 case
        data_loader = DataLoaderBond(dataset, batch_size=2, shuffle=False)
        for graph, labels in data_loader:
            assert np.allclose(labels["value"], np.concatenate(ref_label_energy))
            assert np.allclose(labels["index"], ref_label_index_2)
            if lt:
                assert np.allclose(labels["scaler_mean"], np.concatenate(ref_mean))
                assert np.allclose(labels["scaler_stdev"], np.concatenate(ref_std))

    assert_label(False)
    assert_label(True)


def test_dataloader_reaction():
    ref_label_class = [0, 1]
    ref_num_mols = [2, 3]

    dataset = ElectrolyteReactionDataset(
        grapher=get_grapher_hetero(),
        molecules=test_files.joinpath("electrolyte_struct_rxn_clfn.sdf"),
        labels=test_files.joinpath("electrolyte_label_rxn_clfn.yaml"),
        extra_features=test_files.joinpath("electrolyte_feature_rxn_clfn.yaml"),
        feature_transformer=False,
        label_transformer=False,
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


def test_dataloader_reaction_network():
    ref_label_class = [0, 1]

    dataset = ElectrolyteReactionNetworkDataset(
        grapher=get_grapher_hetero(),
        molecules=test_files.joinpath("electrolyte_struct_rxn_ntwk_clfn.sdf"),
        labels=test_files.joinpath("electrolyte_label_rxn_ntwk_clfn.yaml"),
        extra_features=test_files.joinpath("electrolyte_feature_rxn_ntwk_clfn.yaml"),
        feature_transformer=False,
        label_transformer=False,
    )

    # batch size 1 case (exactly the same as test_dataset)
    data_loader = DataLoaderReactionNetwork(dataset, batch_size=1, shuffle=False)
    for i, (graph, labels) in enumerate(data_loader):
        assert np.allclose(labels["value"], ref_label_class[i])

    # batch size 2 case
    data_loader = DataLoaderReactionNetwork(dataset, batch_size=2, shuffle=False)
    for graph, labels in data_loader:
        assert np.allclose(labels["value"], ref_label_class)
