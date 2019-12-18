import numpy as np
import os
from gnn.data.electrolyte import ElectrolyteDataset
from gnn.data.qm9 import QM9Dataset


test_files = os.path.dirname(__file__)


def test_electrolyte_label():
    ref_label_energies = [[0, 0.1, 0, 0.2, 0, 0.3], [0.4, 0, 0, 0.5, 0, 0]]
    ref_label_indicators = [[0, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0]]

    dataset = ElectrolyteDataset(
        sdf_file=os.path.join(test_files, "EC_struct.sdf"),
        label_file=os.path.join(test_files, "EC_label.txt"),
    )
    size = len(dataset)
    assert size == 2

    for i in range(size):
        _, label = dataset[i]
        assert np.allclose(label["value"], ref_label_energies[i])
        assert np.array_equal(label["indicator"], ref_label_indicators[i])


def test_qm9_label():
    dataset = QM9Dataset(
        sdf_file=os.path.join(test_files, "gdb9_n2.sdf"),
        label_file=os.path.join(test_files, "gdb9_n2.sdf.csv"),
        unit_conversion=False,
        normalize_extensive=False,
    )
    size = len(dataset)
    assert size == 2
    for i in range(size):
        _, label = dataset[i]
        assert np.allclose(
            label,
            [
                157.7118,
                157.70997,
                157.70699,
                0,
                13.21,
                -0.3877,
                0.1171,
                0.5048,
                35.3641,
                0.044749,
                -40.47893,
                -40.476062,
                -40.475117,
                -40.498597,
                6.469,
                -395.999594594,
                -398.643290011,
                -401.014646522,
                -372.471772148,
            ],
        )
        break


def test_qm9_label_mu_alpha():
    dataset = QM9Dataset(
        sdf_file=os.path.join(test_files, "gdb9_n2.sdf"),
        label_file=os.path.join(test_files, "gdb9_n2.sdf.csv"),
        properties=["mu", "alpha"],
        unit_conversion=False,
        normalize_extensive=False,
    )
    size = len(dataset)
    assert size == 2
    for i in range(size):
        _, label = dataset[i]
        assert np.allclose(label, [0, 13.21])
        break
