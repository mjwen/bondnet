import os
import numpy as np
from collections import defaultdict
import pandas as pd
from gnn.analysis.feature_analyzer import (
    PCAAnalyzer,
    TSNEAnalyzer,
    UMAPAnalyzer,
    StdevThreshold,
    PearsonCorrelation,
    plot_heat_map,
    write_dataset_raw_features,
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


def select_data_by_species(feature_file, metadata_file):
    """
    Mask species that are not as dense as others.
    """
    features = pd.read_csv(feature_file, sep="\t", header=None, index_col=None)
    features = features.to_numpy()
    metadata = pd.read_csv(metadata_file, sep="\t", header=0, index_col=None)
    metadata = metadata.to_dict(orient="list")
    metadata = {k: np.asarray(v) for k, v in metadata.items()}

    new_metadata = defaultdict(list)
    keys = metadata.keys()

    # major_species = {"C-H", "C-C", "C-O", "C-F", "H-O"}
    major_species = {"C-H", "C-C", "C-O", "C-F", "H-O", "C-Li", "F-Li", "Li-O"}

    for i, species in enumerate(metadata["species"]):
        if species in major_species:
            for k in keys:
                new_metadata[k].append(metadata[k][i])
        else:
            for k in keys:
                if k == "species":
                    new_metadata[k].append("others")
                else:
                    new_metadata[k].append(metadata[k][i])
    metadata = {k: np.asarray(v) for k, v in new_metadata.items()}

    return features, metadata


def pca_analysis(
    feature_file="~/Applications/db_access/mol_builder/post_analysis/feats.tsv",
    metadata_file="~/Applications/db_access/mol_builder/post_analysis/feats_metadata.tsv",
):
    # analyzer = PCAAnalyzer.from_csv(feature_file, metadata_file, sep="\t")
    # analyzer.compute()

    features, metadata = select_data_by_species(feature_file, metadata_file)
    analyzer = PCAAnalyzer(features, metadata)
    analyzer.compute()

    filename = "~/Applications/db_access/mol_builder/post_analysis/pca_embedding_eng.pdf"
    analyzer.plot_via_umap_points(metadata_key_as_color="energy", filename=filename)
    filename = "~/Applications/db_access/mol_builder/post_analysis/pca_embedding.pdf"
    analyzer.plot_via_umap_points(metadata_key_as_color="species", filename=filename)


def tsne_analysis(
    feature_file="~/Applications/db_access/mol_builder/post_analysis/feats.tsv",
    metadata_file="~/Applications/db_access/mol_builder/post_analysis/feats_metadata.tsv",
):
    # analyzer = TSNEAnalyzer.from_csv(feature_file, metadata_file, sep="\t")
    # analyzer.compute()

    features, metadata = select_data_by_species(feature_file, metadata_file)
    analyzer = TSNEAnalyzer(features, metadata)
    analyzer.compute()

    key = "prediction"
    # key = 'target'
    # key = 'error'
    filename = f"~/Applications/db_access/mol_builder/post_analysis/tsne_{key}.pdf"
    analyzer.plot_via_umap_points(filename, key, False)

    filename = "~/Applications/db_access/mol_builder/post_analysis/tnse_species.pdf"
    analyzer.plot_via_umap_points(filename, "species", True)


def umap_analysis(
    feature_file="~/Applications/db_access/mol_builder/post_analysis/feats.tsv",
    metadata_file="~/Applications/db_access/mol_builder/post_analysis/feats_metadata.tsv",
):

    # analyzer = UMAPAnalyzer.from_csv(feature_file, metadata_file, sep="\t")
    # analyzer.compute()

    features, metadata = select_data_by_species(feature_file, metadata_file)
    analyzer = UMAPAnalyzer(features, metadata)
    analyzer.compute(n_neighbors=200, min_dist=0.9)

    key = "prediction"
    # key = 'target'
    # key = 'error'
    filename = f"~/Applications/db_access/mol_builder/post_analysis/umap_{key}.pdf"
    analyzer.plot_via_umap_points(filename, "prediction", False)

    filename = "~/Applications/db_access/mol_builder/post_analysis/umap_species.pdf"
    analyzer.plot_via_umap_points(filename, "species", True)
    # filename = "~/Applications/db_access/mol_builder/post_analysis/umap_species.html"
    # analyzer.plot_via_umap_interactive(filename, "species", True)

    analyzer.write_embedding_to_csv(
        filename="~/applications/db_access/mol_builder/post_analysis/umap_embedding.tsv",
        sep="\t",
    )


def write_raw_features():
    sdf_file = "~/Applications/db_access/mol_builder/struct_rxn_ntwk_rgrn_qc.sdf"
    label_file = "~/Applications/db_access/mol_builder/label_rxn_ntwk_rgrn_qc.yaml"
    feature_file = "~/Applications/db_access/mol_builder/feature_rxn_ntwk_rgrn_qc.yaml"
    png_dir = "~/Applications/db_access/mol_builder/mol_png_id"
    tex_file = "~/Applications/db_access/mol_builder/tex_raw_features.tex"

    write_dataset_raw_features(sdf_file, label_file, feature_file, png_dir, tex_file)


if __name__ == "__main__":
    # dataset = get_dataset_electrolyte()
    # dataset = get_dataset_electrolyte()
    # dataset = get_dataset_qm9()
    # not_satisfied = feature_stdev(dataset)
    # corelation(dataset, not_satisfied)

    write_raw_features()

    # pca_analysis()
    # tsne_analysis()
    # umap_analysis()
