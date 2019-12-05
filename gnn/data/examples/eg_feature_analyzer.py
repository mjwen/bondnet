import os
import numpy as np
from gnn.data.electrolyte import ElectrolyteDataset
from gnn.data.qm9 import QM9Dataset
from gnn.data.feature_analyzer import StdevThreshold, PearsonCorrelation, plot_heat_map


def get_dataset_electrolyte():
    return ElectrolyteDataset(
        sdf_file="/Users/mjwen/Applications/mongo_db_access/extracted_data/sturct_n200.sdf",
        label_file="/Users/mjwen/Applications/mongo_db_access/extracted_data/label_n200.txt",
    )


def get_dataset_qm9():
    return QM9Dataset(
        sdf_file="/Users/mjwen/Documents/Dataset/qm9/gdb9.sdf",
        label_file="/Users/mjwen/Documents/Dataset/qm9/gdb9.sdf.csv",
    )


def feature_stdev(dataset):
    analyzer = StdevThreshold(dataset, threshold=0.0)
    not_satisfied = {}
    for ntype in ["atom", "bond"]:
        not_satisfied[ntype] = analyzer.compute(ntype)
    return not_satisfied


def corelation(dataset, excludes):
    analyzer = PearsonCorrelation(dataset)

    for ntype in ["atom", "bond"]:
        exclude = excludes[ntype]
        corr = analyzer.compute(ntype, exclude)
        filename = os.path.join(os.path.dirname(__file__), "{}_heatmap.pdf".format(ntype))
        labels = np.delete(dataset.feature_name[ntype], excludes[ntype])
        plot_heat_map(corr, labels, filename)


if __name__ == "__main__":
    dataset = get_dataset_electrolyte()
    # dataset = get_dataset_qm9()
    not_satisfied = feature_stdev(dataset)
    corelation(dataset, not_satisfied)
