import os
import numpy as np
from gnn.data.electrolyte import ElectrolyteDataset
from gnn.data.qm9 import QM9Dataset
from gnn.data.feature_analyzer import (
    StdevThreshold,
    PearsonCorrelation,
    plot_heat_map,
    PCAAnalyzer,
    TSNEAnalyzer,
)


def get_dataset_electrolyte():
    return ElectrolyteDataset(
        sdf_file="~/Applications/mongo_db_access/extracted_data/struct_n200.sdf",
        label_file="~/Applications/mongo_db_access/extracted_data/label_n200.txt",
        # sdf_file="~/Applications/mongo_db_access/extracted_data/struct_charge0.sdf",
        # label_file="~/Applications/mongo_db_access/extracted_data/label_charge0.txt",
        pickle_dataset=False,
    )


def get_pickled_electrolyte():
    return ElectrolyteDataset(
        sdf_file="~/Applications/mongo_db_access/extracted_data/struct_n200.sdf.pkl",
        label_file="~/Applications/mongo_db_access/extracted_data/label_n200.txt.pkl",
    )


def get_dataset_qm9():
    return QM9Dataset(
        sdf_file="~/Documents/Dataset/qm9/gdb9_n200.sdf",
        label_file="~/Documents/Dataset/qm9/gdb9_n200.sdf.csv",
        pickle_dataset=True,
    )


def get_pickled_qm9():
    return QM9Dataset(
        sdf_file="~/Documents/Dataset/qm9/gdb9_n200.sdf.pkl",
        label_file="~/Documents/Dataset/qm9/gdb9_n200.sdf.csv.pkl",
    )


def feature_stdev(dataset):
    analyzer = StdevThreshold(dataset)
    not_satisfied = {}
    for ntype in ["atom", "bond"]:
        not_satisfied[ntype] = analyzer.compute(ntype, threshold=1e-8)
    return not_satisfied


def corelation(dataset, excludes):
    analyzer = PearsonCorrelation(dataset)

    for ntype in ["atom", "bond"]:
        exclude = excludes[ntype]
        corr = analyzer.compute(ntype, exclude)
        filename = os.path.join(os.path.dirname(__file__), "{}_heatmap.pdf".format(ntype))
        labels = np.delete(dataset.feature_name[ntype], excludes[ntype])
        plot_heat_map(corr, labels, filename)


def pca_analysis(dataset):
    analyzer = PCAAnalyzer(dataset)
    analyzer.compute()


def tsne_analysis(dataset):
    analyzer = TSNEAnalyzer(dataset)
    analyzer.compute()


if __name__ == "__main__":
    # dataset = get_dataset_electrolyte()
    # # dataset = get_dataset_qm9()
    # not_satisfied = feature_stdev(dataset)
    # corelation(dataset, not_satisfied)

    # dataset = get_dataset_electrolyte()
    # pca_analysis(dataset)

    dataset = get_dataset_electrolyte()
    tsne_analysis(dataset)
