import numpy as np
import os
from gnn.data.electrolyte import ElectrolyteDataset
from gnn.data.qm9 import QM9Dataset
from gnn.data.dataloader import DataLoaderBond, DataLoaderMolecule

test_files = os.path.dirname(__file__)


# do not assert feature and graph struct, which is handled by dgl,
# and it should be correct
def test_dataloader_electrolyte():
    def assert_label(lt):
        ref_label_energies = [[0, 0.1, 0, 0.2, 0, 0.3], [0.4, 0, 0, 0.5, 0, 0]]
        ref_label_indicators = [[0, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0]]

        if lt:
            mean = np.mean(ref_label_energies)
            std = np.std(ref_label_energies)
            ref_label_energies = [
                (np.asarray(a) - mean) / std for a in ref_label_energies
            ]
            ref_scales = [[std] * len(x) for x in ref_label_energies]

        dataset = ElectrolyteDataset(
            sdf_file=os.path.join(test_files, "EC_struct.sdf"),
            label_file=os.path.join(test_files, "EC_label.txt"),
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


# do not assert feature and graph struct, which is handled by dgl,
# and it should be correct
def test_dataloader_qm9():
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
