import os
import numpy as np
from gnn.data.dataset import ElectrolyteDataset
from gnn.data.feature_analyzer import StdevThreshold, PearsonCorrelation, plot_heat_map


def get_dataset():
    return ElectrolyteDataset(
        sdf_file="/Users/mjwen/Applications/mongo_db_access/extracted_data/sturct_n200.sdf",
        label_file="/Users/mjwen/Applications/mongo_db_access/extracted_data/label_n200.txt",
    )


def feature_stdev():
    dataset = get_dataset()
    analyzer = StdevThreshold(dataset, threshold=0.0)
    not_satisfied = {}
    for ntype in ["atom", "bond"]:
        not_satisfied[ntype] = analyzer.compute(ntype)
    return not_satisfied


def corelation(excludes):
    dataset = get_dataset()
    analyzer = PearsonCorrelation(dataset)

    for ntype in ["atom", "bond"]:
        exclude = excludes[ntype]
        corr = analyzer.compute(ntype, exclude)
        filename = os.path.join(os.path.dirname(__file__), "{}_heatmap.pdf".format(ntype))
        labels = np.delete(dataset.feature_name[ntype], excludes[ntype])
        plot_heat_map(corr, labels, filename)


if __name__ == "__main__":
    not_satisfied = feature_stdev()
    corelation(not_satisfied)
