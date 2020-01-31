import os
import glob
import numpy as np
from collections import defaultdict
from gnn.data.electrolyte import ElectrolyteBondDataset
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
from gnn.utils import expand_path
from gnn.data.utils import TexWriter


def get_dataset_electrolyte():
    dataset = ElectrolyteBondDataset(
        # sdf_file="~/Applications/mongo_db_access/extracted_mols/struct.sdf",
        # label_file="~/Applications/mongo_db_access/extracted_mols/label.txt",
        # feature_file="~/Applications/mongo_db_access/extracted_mols/feature.yaml",
        # sdf_file="~/Applications/mongo_db_access/extracted_mols/struct_n200.sdf",
        # label_file="~/Applications/mongo_db_access/extracted_mols/label_n200.txt",
        # feature_file="~/Applications/mongo_db_access/extracted_mols/feature_n200.yaml",
        sdf_file="~/Applications/mongo_db_access/extracted_mols/struct_charge0.sdf",
        label_file="~/Applications/mongo_db_access/extracted_mols/label_charge0.txt",
        feature_file="~/Applications/mongo_db_access/extracted_mols/feature_charge0.yaml",
        # sdf_file="~/Applications/mongo_db_access/extracted_mols/struct_charge0_CC.sdf",
        # label_file="~/Applications/mongo_db_access/extracted_mols/label_charge0_CC.txt",
        # feature_file="~/Applications/mongo_db_access/extracted_mols/feature_charge0_CC.yaml",
        atom_featurizer_with_extra_info=True,
        bond_length_featurizer="bin",
        pickle_dataset=False,
    )

    print("@@@ len(dataset)", len(dataset))
    return dataset


def get_pickled_electrolyte():
    return ElectrolyteBondDataset(
        sdf_file="~/Applications/mongo_db_access/extracted_mols/struct_n200.sdf.pkl",
        label_file="~/Applications/mongo_db_access/extracted_mols/label_n200.txt.pkl",
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
    for ntype in ["atom", "bond", "global"]:
        not_satisfied[ntype] = analyzer.compute(ntype, threshold=1e-8)
    return not_satisfied


def corelation(dataset, excludes):
    analyzer = PearsonCorrelation(dataset)

    for ntype in ["atom", "bond", "global"]:
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


def kmeans_analysis(
    label_file="~/Applications/mongo_db_access/extracted_mols/label_charge0_CC.txt",
    sdf_file="~/Applications/mongo_db_access/extracted_mols/struct_charge0_CC.sdf",
    feature_file="~/Applications/mongo_db_access/extracted_mols/feature_charge0_CC.yaml",
    png_dir="~/Applications/mongo_db_access/extracted_mols/mol_png",
    tex_file="~/Applications/mongo_db_access/kmeans_cluster.tex",
):
    # kmeans analysis
    dataset = ElectrolyteBondDataset(
        sdf_file, label_file, feature_file, label_transformer=False
    )
    analyzer = KMeansAnalyzer(dataset)
    features, clusters, centers, energies = analyzer.compute()
    classes = defaultdict(list)
    for i, c in enumerate(clusters):
        classes[c].append(i)

    labels = read_label(filename=label_file, sort_by_formula=False)
    structs = read_sdf(filename=sdf_file)
    all_pngs = glob.glob(os.path.join(expand_path(png_dir), "*.png"))

    tex_file = expand_path(tex_file)
    with open(tex_file, "w") as f:
        f.write(TexWriter.head())

        for c, cen in enumerate(centers):

            ### write cluster info
            f.write("\n" * 2 + r"\newpage" + "\n")
            f.write(r"\begin{verbatim}" + "\n")

            f.write("center: ")
            for j in cen:
                f.write("{:12.5e} ".format(j))
            f.write("\n")

            cls = classes[c]
            f.write("# bond index     energy\n")
            for j in cls:
                f.write("{:<4d}     {:12.5e}\n".format(j, energies[j]))
            f.write("\n" * 3)
            f.write(r"\end{verbatim}" + "\n")

            ### write struct, label, and mol figures
            for j in cls:
                rxn = labels[j]
                reactant = rxn["reactants"]
                products = rxn["products"]
                raw = rxn["raw"]

                f.write("\n" * 2 + r"\newpage" + "\n")
                f.write(r"\begin{verbatim}" + "\n")

                # sdf info
                f.write(structs[reactant])

                # label info
                f.write("\n" * 2)
                length = 80  # split it int chunks of fixed length
                raw = [raw[0 + i : length + i] for i in range(0, len(raw), length)]
                for r in raw:
                    f.write(r + "\n")

                f.write(r"\end{verbatim}" + "\n")

                # figure
                filename = None
                for name in all_pngs:
                    if reactant in name:
                        filename = name
                        break
                if filename is None:
                    raise Exception(
                        "cannot find png file for {} in {}".format(reactant, png_dir)
                    )
                f.write(TexWriter.single_figure(filename))

                f.write(r"\begin{equation*}\downarrow\end{equation*}")

                for i, p in enumerate(products):
                    if i > 0:
                        f.write(r"\begin{equation*}+\end{equation*}")
                    filename = None
                    for name in all_pngs:
                        if p in name:
                            filename = name
                            break
                    if filename is None:
                        raise Exception(
                            "cannot find png file for {} in {}".format(p, png_dir)
                        )
                    f.write(TexWriter.single_figure(filename))

        # tail
        tail = r"\end{document}"
        f.write(tail)


def write_features(
    sdf_file="~/Applications/mongo_db_access/extracted_mols/struct.sdf",
    label_file="~/Applications/mongo_db_access/extracted_mols/label.txt",
    feature_file="~/Applications/mongo_db_access/extracted_mols/feature.yaml",
    # sdf_file="~/Applications/mongo_db_access/extracted_mols/struct_n200.sdf",
    # label_file="~/Applications/mongo_db_access/extracted_mols/label_n200.txt",
    # feature_file="~/Applications/mongo_db_access/extracted_mols/feature_n200.yaml",
    # sdf_file="~/Applications/mongo_db_access/extracted_mols/struct_charge0.sdf",
    # label_file="~/Applications/mongo_db_access/extracted_mols/label_charge0.txt",
    # sdf_file="~/Applications/mongo_db_access/extracted_mols/struct_charge0_CC.sdf",
    # label_file="~/Applications/mongo_db_access/extracted_mols/label_charge0_CC.txt",
    png_dir="~/Applications/mongo_db_access/extracted_mols/mol_png",
    tex_file="~/Applications/mongo_db_access/tex_features.tex",
):

    dataset = ElectrolyteBondDataset(
        sdf_file,
        label_file,
        feature_file,
        feature_transformer=False,
        label_transformer=False,
        atom_featurizer_with_extra_info=True,
        bond_length_featurizer="bin",
    )

    structs = read_sdf(filename=sdf_file)
    labels = read_label(label_file, sort_by_formula=False)
    png_dir = expand_path(png_dir)
    all_pngs = glob.glob(os.path.join(png_dir, "*.png"))

    tex_file = expand_path(tex_file)
    with open(tex_file, "w") as f:
        f.write(TexWriter.head())

        for i, (g, _, _) in enumerate(dataset):

            # we use g.graph_id instead of i here because some sdf entries may fail
            # when creating the dataset, and thus there could be mismatch
            rxn = labels[g.graph_id]

            reactant = rxn["reactants"]
            raw = rxn["raw"]

            f.write(TexWriter.newpage())
            # sdf info
            f.write(TexWriter.verbatim(structs[reactant]))
            # label info (raw)
            f.write(TexWriter.verbatim(TexWriter.resize_string(raw)))

            # molecule figure
            filename = None
            for name in all_pngs:
                if reactant in name:
                    filename = name
                    break
            if filename is None:
                raise Exception(
                    "cannot find png file for {} in {}".format(reactant, png_dir)
                )
            f.write(TexWriter.single_figure(filename))

            # feature info
            # atom feature
            f.write("atom feature:\n")
            ft = g.nodes["atom"].data["feat"]
            # ft = np.asarray(ft, dtype=np.int32)  # they are actually int feature
            header = dataset.feature_name["atom"]
            tables = TexWriter.beautifultable(
                ft,
                header,
                first_column=[1 + i for i in range(len(ft))],
                first_column_header="id",
                num_tables=1,
            )
            f.write(TexWriter.verbatim(tables))

            # bond feature
            f.write("\n\nbond feature:\n")
            ft = g.nodes["bond"].data["feat"]
            # ft = np.asarray(ft, dtype=np.int32)  # they are actually int feature
            header = dataset.feature_name["bond"]
            tables = TexWriter.beautifultable(
                ft,
                header,
                first_column=[1 + i for i in range(len(ft))],
                first_column_header="id",
                num_tables=1,
            )
            f.write(TexWriter.verbatim(tables))

            # global feature
            f.write("\n\nglobal feature:\n")
            ft = g.nodes["global"].data["feat"]
            # ft = np.asarray(ft, dtype=np.int32)  # they are actually int feature
            header = dataset.feature_name["global"]
            tables = TexWriter.beautifultable(
                ft,
                header,
                first_column=[1 + i for i in range(len(ft))],
                first_column_header="id",
                num_tables=1,
            )
            f.write(TexWriter.verbatim(tables))

        f.write(TexWriter.tail())


if __name__ == "__main__":
    dataset = get_dataset_electrolyte()
    # dataset = get_dataset_qm9()
    not_satisfied = feature_stdev(dataset)
    corelation(dataset, not_satisfied)

    # dataset = get_dataset_electrolyte()
    # pca_analysis(dataset)
    # tsne_analysis(dataset)

    # write_features()
    # kmeans_analysis()
