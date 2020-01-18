import os
import numpy as np
from collections import defaultdict
from gnn.data.electrolyte import ElectrolyteDataset
from gnn.data.qm9 import QM9Dataset
from gnn.data.feature_analyzer import (
    StdevThreshold,
    PearsonCorrelation,
    plot_heat_map,
    PCAAnalyzer,
    TSNEAnalyzer,
    KMeansAnalyzer,
    read_label,
    read_sdf,
)


def get_dataset_electrolyte():
    return ElectrolyteDataset(
        sdf_file="~/Applications/mongo_db_access/extracted_data/struct_n200.sdf",
        label_file="~/Applications/mongo_db_access/extracted_data/label_n200.txt",
        # sdf_file="~/Applications/mongo_db_access/extracted_data/struct_charge0.sdf",
        # label_file="~/Applications/mongo_db_access/extracted_data/label_charge0.txt",
        # sdf_file="~/Applications/mongo_db_access/extracted_data/struct_charge0_CC.sdf",
        # label_file="~/Applications/mongo_db_access/extracted_data/label_charge0_CC.txt",
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


def kmeans_analysis(cluster_info_file="cluster_info.txt"):
    # sdf_file = "~/Applications/mongo_db_access/extracted_data/struct_n200.sdf"
    # label_file = "~/Applications/mongo_db_access/extracted_data/label_n200.txt"
    # sdf_file="~/Applications/mongo_db_access/extracted_data/struct_charge0.sdf"
    # label_file="~/Applications/mongo_db_access/extracted_data/label_charge0.txt"
    sdf_file = "~/Applications/mongo_db_access/extracted_data/struct_charge0_CC.sdf"
    label_file = "~/Applications/mongo_db_access/extracted_data/label_charge0_CC.txt"

    # kmeans analysis
    dataset = ElectrolyteDataset(sdf_file, label_file, label_transformer=False)
    analyzer = KMeansAnalyzer(dataset)
    features, clusters, centers, energies = analyzer.compute()

    # data and label
    # structs = read_sdf(sdf_file)
    labels = read_label(label_file, sort_by_formula=False)

    # output file 1, cluster info, i.e. which bond are classified to the bond
    classes = defaultdict(list)
    for i, c in enumerate(clusters):
        classes[c].append(i)

    with open(cluster_info_file, "w") as f:
        f.write("# bonds     energy     raw\n")
        for c, cen in enumerate(centers):
            f.write("center: ")
            for j in cen:
                f.write("{:12.5e} ".format(j))
            f.write("\n")

            cls = classes[c]
            for j in cls:
                f.write(
                    "{:<4d}     {:12.5e}     {}\n".format(
                        j, energies[j], labels[j]["raw"]
                    )
                )
            f.write("\n" * 3)


if __name__ == "__main__":
    # dataset = get_dataset_electrolyte()
    # # dataset = get_dataset_qm9()
    # not_satisfied = feature_stdev(dataset)
    # corelation(dataset, not_satisfied)

    # dataset = get_dataset_electrolyte()
    # pca_analysis(dataset)
    # tsne_analysis(dataset)

    kmeans_analysis()
