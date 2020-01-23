import os
import glob
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
from gnn.utils import expand_path
from beautifultable import BeautifulTable


def get_dataset_electrolyte():
    return ElectrolyteDataset(
        # sdf_file="~/Applications/mongo_db_access/extracted_data/struct_n200.sdf",
        # label_file="~/Applications/mongo_db_access/extracted_data/label_n200.txt",
        # sdf_file="~/Applications/mongo_db_access/extracted_data/struct_charge0.sdf",
        # label_file="~/Applications/mongo_db_access/extracted_data/label_charge0.txt",
        sdf_file="~/Applications/mongo_db_access/extracted_data/struct_charge0_CC.sdf",
        label_file="~/Applications/mongo_db_access/extracted_data/label_charge0_CC.txt",
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


def write_features(
    sdf_file="~/Applications/mongo_db_access/extracted_data/struct.sdf",
    label_file="~/Applications/mongo_db_access/extracted_data/label.txt",
    # sdf_file="~/Applications/mongo_db_access/extracted_data/struct_n200.sdf",
    # label_file="~/Applications/mongo_db_access/extracted_data/label_n200.txt",
    # sdf_file="~/Applications/mongo_db_access/extracted_data/struct_charge0.sdf",
    # label_file="~/Applications/mongo_db_access/extracted_data/label_charge0.txt",
    # sdf_file="~/Applications/mongo_db_access/extracted_data/struct_charge0_CC.sdf",
    # label_file="~/Applications/mongo_db_access/extracted_data/label_charge0_CC.txt",
    png_dir="~/Applications/mongo_db_access/extracted_data/mol_png_cropped",
    tex_file="~/Applications/mongo_db_access/features.tex",
):
    def plot_fig(filename):
        s = r"""
    \begin{center}
    \includegraphics[width=0.4\columnwidth]{"""
        s += filename
        s += r"""}
    \end{center}
            """
        return s

    def fix_length(s, length=80, starter=""):
        """
        Given a string `s` returns a list of string of fixed length.
        """
        return [starter + s[0 + i : length + i] for i in range(0, len(s), length)]

    def two_d_array_to_beautifultable(
        array, header, first_column=None, first_column_header=None, num_tables=2
    ):
        """
        Convert a 2D array to a beautiful table, with the ability to separate the table
        into multiple ones in case there are too many columns.

        first_column: 1D array
        first_column_header: str
        num_tabels: number of tables to split the array (along column)

        Returns:
            a list of beautiful tabels
        """
        array = np.asarray(array)

        tables = []
        headers = []
        arrays = []
        column_width = int(np.ceil(len(array[0]) / num_tables))
        for i in range(num_tables):
            tables.append(BeautifulTable(max_width=80))

            if first_column_header is not None:
                headers.append(
                    [first_column_header]
                    + header[i * column_width : (i + 1) * column_width]
                )
            else:
                headers.append(header[i * column_width : (i + 1) * column_width])

            a = array[:, i * column_width : (i + 1) * column_width]
            if first_column is not None:
                fc = np.atleast_2d(first_column).T
                a = np.concatenate((fc, a), axis=1)
            arrays.append(a)

        for t, h, a in zip(tables, headers, arrays):
            t.column_headers = h
            for row in a:
                t.append_row(row)

        return tables

    # header
    header = r"""
    \documentclass[11pt]{article}

    \usepackage[top=1in, bottom=1in, left=1in, right=1in]{geometry}
    \usepackage{amsmath}
    \usepackage{graphicx}
    \usepackage{caption}
    \usepackage{subcaption}
    \usepackage{color}
    \usepackage{float}    %\begin{figure}[H]

    \begin{document}
    """

    dataset = ElectrolyteDataset(
        sdf_file, label_file, feature_transformer=False, label_transformer=False
    )

    structs = read_sdf(filename=sdf_file)
    labels = read_label(label_file, sort_by_formula=False)
    all_pngs = glob.glob(os.path.join(expand_path(png_dir), "*.png"))

    tex_file = expand_path(tex_file)
    with open(tex_file, "w") as f:
        f.write(header)

        for i, (g, _, _) in enumerate(dataset):

            rxn = labels[i]
            reactant = rxn["reactants"]
            raw = rxn["raw"]

            f.write("\n" * 2 + r"\newpage" + "\n")
            f.write(r"\begin{verbatim}" + "\n")

            # sdf info
            f.write(structs[reactant])

            # label info (raw)
            f.write("\n" * 2)
            for r in fix_length(raw):
                f.write(r + "\n")

            f.write(r"\end{verbatim}" + "\n")

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
            s = plot_fig(filename)
            f.write(s)

            # feature info
            f.write(r"\begin{verbatim}" + "\n")

            # atom feature
            f.write("atom feature:\n")
            ft = g.nodes["atom"].data["feat"]
            ft = np.asarray(ft, dtype=np.int32)  # they are actually int feature
            header = dataset.feature_name["atom"]
            tables = two_d_array_to_beautifultable(
                ft,
                header,
                first_column=[1 + i for i in range(len(ft))],
                first_column_header="id",
                num_tables=2,
            )
            for table in tables:
                for line in str(table):
                    f.write(line)
                f.write("\n")

            # bond feature
            f.write("\n\nbond feature:\n")
            ft = g.nodes["bond"].data["feat"]
            ft = np.asarray(ft, dtype=np.int32)  # they are actually int feature
            header = dataset.feature_name["bond"]
            tables = two_d_array_to_beautifultable(
                ft,
                header,
                first_column=[1 + i for i in range(len(ft))],
                first_column_header="id",
                num_tables=1,
            )
            for table in tables:
                for line in str(table):
                    f.write(line)
                f.write("\n")

            f.write(r"\end{verbatim}" + "\n")

        # tail
        tail = r"\end{document}"
        f.write(tail)


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

    # write cluster info, i.e. which bond are clustered to be close to each other
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


def kmeans_analysis_create_tex(
    # label_file="~/Applications/mongo_db_access/extracted_data/label_n200.txt",
    # sdf_file="~/Applications/mongo_db_access/extracted_data/struct_n200.sdf",
    label_file="~/Applications/mongo_db_access/extracted_data/label_charge0_CC.txt",
    sdf_file="~/Applications/mongo_db_access/extracted_data/struct_charge0_CC.sdf",
    png_dir="~/Applications/mongo_db_access/extracted_data/mol_png_cropped",
    tex_file="~/Applications/mongo_db_access/kmeans_cluster.tex",
):
    def plot_fig(filename):
        s = r"""
\begin{center}
\includegraphics[width=0.4\columnwidth]{"""
        s += filename
        s += r"""}
\end{center}
        """
        return s

    # header
    header = r"""
\documentclass[11pt]{article}
\usepackage[top=1in, bottom=1in, left=1in, right=1in]{geometry}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{color}
\usepackage{float}    %\begin{figure}[H]


\begin{document}
"""
    # kmeans analysis
    dataset = ElectrolyteDataset(sdf_file, label_file, label_transformer=False)
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
        f.write(header)

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
                s = plot_fig(filename)
                f.write(s)

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
                    s = plot_fig(filename)
                    f.write(s)

        # tail
        tail = r"\end{document}"
        f.write(tail)


if __name__ == "__main__":
    # dataset = get_dataset_electrolyte()
    # # dataset = get_dataset_qm9()
    # not_satisfied = feature_stdev(dataset)
    # corelation(dataset, not_satisfied)

    # dataset = get_dataset_electrolyte()
    # pca_analysis(dataset)
    # tsne_analysis(dataset)

    # kmeans_analysis()
    # kmeans_analysis_create_tex()

    write_features()
