import numpy as np
import os
from gnn.data.electrolyte import ElectrolyteDataset
from gnn.data.qm9 import QM9Dataset


ref_label_energies = [[0, 0.1, 0, 0.2, 0, 0.3], [0.4, 0, 0, 0.5, 0, 0]]
ref_label_indicators = [[0, 1, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0]]


def test_electrolyte_label():
    test_files = os.path.dirname(__file__)
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


#
# def test_qm9_label():
#     dataset = QM9Dataset(
#         sdf_file="/Users/mjwen/Documents/Dataset/qm9/gdb9.sdf",
#         label_file="/Users/mjwen/Documents/Dataset/qm9/gdb9.sdf.csv",
#     )
#     size = len(dataset)
#     assert size == 133885
