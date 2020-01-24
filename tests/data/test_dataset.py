import numpy as np
import os
from gnn.data.electrolyte import ElectrolyteDataset
from gnn.data.qm9 import QM9Dataset
from gnn.data.electrolyte_mols import ElectrolyteMoleculeDataset


test_files = os.path.dirname(__file__)


def test_electrolyte_label():
    def assert_label(lt):
        ref_label_energies = [[0, 0.1, 0, 0.2, 0, 0.3], [0.4, 0, 0, 0.5, 0, 0]]
        ref_label_indicators = [[0, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0]]

        if lt:
            mean = np.mean(ref_label_energies)
            std = np.std(ref_label_energies)
            ref_label_energies = [
                (np.asarray(a) - mean) / std for a in ref_label_energies
            ]
            ref_ts = [[std] * len(x) for x in ref_label_energies]

        dataset = ElectrolyteDataset(
            sdf_file=os.path.join(test_files, "EC_struct.sdf"),
            label_file=os.path.join(test_files, "EC_label.txt"),
            feature_file=os.path.join(test_files, "EC_feature.yaml"),
            feature_transformer=False,
            label_transformer=lt,
        )

        size = len(dataset)
        assert size == 2

        for i in range(size):
            _, label, ts = dataset[i]
            assert np.allclose(label["value"], ref_label_energies[i])
            assert np.array_equal(label["indicator"], ref_label_indicators[i])

            if lt:
                assert np.allclose(ts, ref_ts[i])
            else:
                assert ts is None

    assert_label(True)
    assert_label(False)


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
            sdf_file=os.path.join(test_files, "gdb9_n2.sdf"),
            label_file=os.path.join(test_files, "gdb9_n2.sdf.csv"),
            properties=["homo", "u0"],  # homo is intensive and u0 is extensive
            unit_conversion=False,
            feature_transformer=True,
            label_transformer=lt,
        )

        size = len(dataset)
        assert size == 2

        for i in range(size):
            _, label, ts = dataset[i]
            assert np.allclose(label, ref_labels[i])

            if lt:
                assert np.allclose(ts, ref_ts[i])
            else:
                assert ts is None

    assert_label(False)
    assert_label(True)


def test_electrolyte_molecule_label():
    def assert_label(lt):
        ref_labels = np.asarray([[-0.941530613939904], [-8.91357537335352]])
        natoms = np.asarray([[2], [5]])

        if lt:
            ref_labels /= natoms
            ref_ts = natoms

        dataset = ElectrolyteMoleculeDataset(
            sdf_file=os.path.join(test_files, "electrolyte_mols_struct.sdf"),
            label_file=os.path.join(test_files, "electrolyte_mols_label.csv"),
            grapher="homo_bidirected",  # avoid providing feature_file needed by `hetero`
            properties=["atomization_energy"],
            unit_conversion=False,
            feature_transformer=True,
            label_transformer=lt,
        )

        size = len(dataset)
        assert size == 2

        for i in range(size):
            _, label, ts = dataset[i]
            assert np.allclose(label, ref_labels[i])

            if lt:
                assert np.allclose(ts, ref_ts[i])
            else:
                assert ts is None

    assert_label(False)
    assert_label(True)
