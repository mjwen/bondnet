import numpy as np
import os
from gnn.data.dataset import ElectrolyteDataset

test_files = os.path.join(os.path.dirname(__file__), "test_files")

ref_label_indicators = [[0, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0]]
ref_label_energies = [[0, 0.1, 0, 0.2, 0, 0.3], [0.4, 0, 0, 0.5, 0, 0]]


def test_dataset():
    dataset = ElectrolyteDataset(
        sdf_file=os.path.join(test_files, "EC_struct.sdf"),
        label_file=os.path.join(test_files, "EC_label.txt"),
    )
    size = len(dataset)
    assert size == 2

    for i in range(size):
        _, label = dataset[i]
        assert np.allclose(label["energies"], ref_label_energies[i])
        assert np.array_equal(label["indicators"], ref_label_indicators[i])
