import os
from gnn.data.dataset import ElectrolyteDataset
from gnn.data.feature_analyzer import VarianceThreshold, PearsonCorrelation, plot_heat_map


def get_dataset():
    return ElectrolyteDataset(
        sdf_file="/Users/mjwen/Applications/mongo_db_access/extracted_data/sturct_n200.sdf",
        label_file="/Users/mjwen/Applications/mongo_db_access/extracted_data/label_n200.txt",
    )


def feature_variance():
    dataset = get_dataset()
    analyzer = VarianceThreshold(dataset, threshold=0.0)
    for ntype in ["atom", "bond"]:
        analyzer.compute(ntype)


def corelation():
    dataset = get_dataset()
    analyzer = PearsonCorrelation(dataset)

    for ntype in ["atom", "bond"]:
        if ntype == "atom":
            exclude = [9]
        elif ntype == "bond":
            exclude = [3, 4, 5]
        else:
            exclude = None
        corr = analyzer.compute(ntype, exclude)
        filename = os.path.join(os.path.dirname(__file__), "{}_heatmap.pdf".format(ntype))
        plot_heat_map(corr, filename)


if __name__ == "__main__":
    feature_variance()
    corelation()
