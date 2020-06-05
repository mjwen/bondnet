import os
import abc
import glob
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from umap import UMAP
import matplotlib as mpl
import matplotlib.pyplot as plt
from gnn.layer.readout import ConcatenateMeanMax
from gnn.analysis import umap_plot
from gnn.analysis.utils import TexWriter
from gnn.utils import expand_path, yaml_load
from rdkit import Chem
from gnn.data.utils import get_dataset_species
from gnn.data.featurizer import (
    AtomFeaturizerMinimum,
    AtomFeaturizerFull,
    BondAsNodeFeaturizerMinimum,
    BondAsNodeFeaturizerFull,
    GlobalFeaturizer,
)
from gnn.data.grapher import HeteroMoleculeGraph


def write_dataset_raw_features_to_tex(
    sdf_file, label_file, feature_file, png_dir, tex_file
):
    def get_sdfs(fname):
        structs = []
        with open(expand_path(fname), "r") as f:
            for line in f:
                if "index" in line:
                    body = line
                elif "$$$$" in line:
                    structs.append(body)
                else:
                    body += line
        return structs

    def get_molecules(fname):
        supp = Chem.SDMolSupplier(expand_path(fname), sanitize=True, removeHs=False)
        molecules = [m for m in supp]
        return molecules

    # def get_label(fname):
    #     return yaml_load(fname)
    #
    def get_extra_features(fname):
        return yaml_load(fname)

    def get_grapher():
        # atom_featurizer = AtomFeaturizerFull()
        # bond_featurizer = BondAsNodeFeaturizerFull(length_featurizer=None, dative=False)
        # global_featurizer = GlobalFeaturizer(allowed_charges=None)

        atom_featurizer = AtomFeaturizerMinimum()
        bond_featurizer = BondAsNodeFeaturizerMinimum(length_featurizer=None)
        global_featurizer = GlobalFeaturizer(allowed_charges=[-1, 0, 1])

        grapher = HeteroMoleculeGraph(
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            self_loop=True,
        )
        return grapher

    sdfs = get_sdfs(sdf_file)

    molecules = get_molecules(sdf_file)
    species = get_dataset_species(molecules)

    # labels = get_label(label_file)
    extra_features = get_extra_features(feature_file)

    grapher = get_grapher()

    png_dir = expand_path(png_dir)
    all_pngs = glob.glob(os.path.join(png_dir, "*.png"))

    tex_file = expand_path(tex_file)

    with open(tex_file, "w") as f:
        f.write(TexWriter.head())

        for i, m in enumerate(molecules):
            if m is None:
                continue

            g = grapher.build_graph_and_featurize(
                m, extra_feats_info=extra_features[i], dataset_species=species
            )

            mol_id = sdfs[i].strip().split("_")[0]

            f.write(TexWriter.newpage())

            # sdf info
            f.write(TexWriter.verbatim(sdfs[i]))

            # molecule figure
            for name in all_pngs:
                if mol_id in name:
                    filename = name
                    break
            else:
                raise Exception(
                    "cannot find png file for {} in {}".format(mol_id, png_dir)
                )
            f.write(TexWriter.single_figure(filename))

            # feature info
            # atom feature
            f.write("atom feature:\n")
            ft = g.nodes["atom"].data["feat"]
            ft = np.asarray(ft, dtype=np.int32)  # they are actually int feature
            header = grapher.feature_name["atom"]
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
            ft = np.asarray(ft, dtype=np.int32)  # they are actually int feature
            header = grapher.feature_name["bond"]
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
            ft = np.asarray(ft, dtype=np.int32)  # they are actually int feature
            header = grapher.feature_name["global"]
            tables = TexWriter.beautifultable(
                ft,
                header,
                first_column=[1 + i for i in range(len(ft))],
                first_column_header="id",
                num_tables=1,
            )
            f.write(TexWriter.verbatim(tables))

        f.write(TexWriter.tail())


class FeatureAggregator:
    def __init__(self, dataset):
        self.dataset = dataset

    def _stack_feature(self, ntype):
        feature = []
        for g, _, _ in self.dataset:
            feature.append(g.nodes[ntype].data["feat"])
        return np.concatenate(feature)

    def _stack_feature_and_label(self, ntype="bond"):
        """
        Stack feature (each feature is the bond feature and concatenated with the mean
        and max of the features of the atoms constituting the bond) and label whose
        corresponding label indicator is True, i.e. we have energy for the label.
        """
        features = []
        labels = []
        for g, lb, _ in self.dataset:
            # indices of bond that has energy
            indices = [int(i) for i, v in enumerate(lb["indicator"]) if v == 1]
            labels.append(lb["value"][indices])
            features.append(g.nodes[ntype].data["feat"][indices])

        features = np.concatenate(features)
        labels = np.concatenate(labels)

        return features, labels

    def _stack_bond_feature_plus_atom_feature_and_label(self):
        """
        Stack feature and label whose corresponding label indicator is True, i.e. we
        have energy for the label.
        """
        features = []
        labels = []

        for g, lb, _ in self.dataset:

            # indices of bond that has energy
            indices = [int(i) for i, v in enumerate(lb["indicator"]) if v == 1]
            labels.append(lb["value"][indices])

            all_feats = {
                "atom": g.nodes["atom"].data["feat"],
                "bond": g.nodes["bond"].data["feat"],
                "global": g.nodes["global"].data["feat"],
            }

            etypes = [("atom", "a2b", "bond")]
            layer = ConcatenateMeanMax(etypes)
            rst = layer(g, all_feats)
            bond_feats = rst["bond"][indices]
            features.append(bond_feats)

        features = np.concatenate(features)
        labels = np.concatenate(labels)

        return features, labels


class BaseAnalyzer(abc.ABC):
    """
    Base analyzer for features.

    Args:
        features (2D array): each row is the features of a data point.
        metadata (dict): metadata (e.g. label) of each data point, with string of key
            and 1D array (should have the same size as `features`) as list.
    """

    def __init__(self, features, metadata=None):
        self.features = features
        self.metadata = metadata
        self.embedding = None
        self.model = None

    @classmethod
    def from_csv(cls, feature_file, metadata_file, sep=","):
        """
        Read csv files. feature_file should have neither header and row
        index and metadata_file should have header but not row index.

        For example, see: http://projector.tensorflow.org
        """
        features = pd.read_csv(feature_file, sep=sep, header=None, index_col=None)
        features = features.to_numpy()
        metadata = pd.read_csv(metadata_file, sep=sep, header=0, index_col=None)
        metadata = metadata.to_dict(orient="list")
        metadata = {k: np.asarray(v) for k, v in metadata.items()}

        return cls(features, metadata)

    def write_embedding_to_csv(self, filename, sep=","):
        if self.embedding is not None:
            df = pd.DataFrame(
                {"1st_dim": self.embedding[:, 0], "2nd_dim": self.embedding[:, 1]}
            )
            df.to_csv(filename, sep=sep, header=True, index=False)
        else:
            raise RuntimeError("`embedding` is None, call `compute()` to get it first")

    @abc.abstractmethod
    def compute(self, **kwargs):
        raise NotImplementedError

    def plot(self, metadata_key_as_color="energy", filename="embedding.pdf"):
        plot_scatter([self.embedding], [self.metadata[metadata_key_as_color]], filename)

    def plot_via_umap_points(
        self,
        filename="embedding.pdf",
        metadata_key_as_color="energy",
        categorical_color=False,
    ):
        if not categorical_color:
            umap_plot.points(
                expand_path(filename),
                self.embedding,
                values=self.metadata[metadata_key_as_color],
                theme="viridis",
            )
        else:
            umap_plot.points(
                expand_path(filename),
                self.embedding,
                labels=self.metadata[metadata_key_as_color],
                color_key_cmap="tab20",
            )

    def plot_via_umap_interactive(
        self,
        filename="embedding.html",
        metadata_key_as_color="energy",
        categorical_color=False,
    ):
        if categorical_color:
            umap_plot.interactive(
                expand_path(filename),
                self.model,
                values=self.metadata[metadata_key_as_color],
                hover_data=pd.DataFrame(self.metadata),
                theme="viridis",
                point_size=5,
            )
        else:
            umap_plot.interactive(
                expand_path(filename),
                self.model,
                labels=self.metadata[metadata_key_as_color],
                hover_data=pd.DataFrame(self.metadata),
                color_key_cmap="Paired",
                point_size=5,
            )


class PCAAnalyzer(BaseAnalyzer):
    """
    PCA analysis for features.
    """

    def compute(self, n_components=2, **kwargs):
        self.model = PCA(n_components=n_components, **kwargs)
        embedding = self.model.fit_transform(self.features)
        self.embedding = embedding

        return self.embedding


class TSNEAnalyzer(BaseAnalyzer):
    """
    TSNE analysis for features.
    """

    def compute(self, n_components=2, perplexity=30, init="pca", verbose=2, **kwargs):
        self.model = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            init=init,
            verbose=verbose,
            **kwargs,
        )
        embedding = self.model.fit_transform(self.features)
        self.embedding = embedding

        return self.embedding


class UMAPAnalyzer(BaseAnalyzer):
    """
    UMAP analysis for features.
    """

    def compute(
        self,
        n_neighbors=100,
        n_components=2,
        min_dist=0.5,
        metric="euclidean",
        verbose=True,
        n_epochs=1000,
        **kwargs
    ):
        self.model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
            verbose=True,
            n_epochs=n_epochs,
            **kwargs,
        )
        embedding = self.model.fit_transform(self.features)
        self.embedding = embedding

        return self.embedding


class StdevThreshold(FeatureAggregator):
    """
    Analyzer to check the standard deviation of the features.

    Args:
        dataset: a :class:`gnn.data.dataset.ElectrolyteData` dataset.
    """

    def compute(self, ntype, threshold=1e-8):
        """
        Compute the stdandard deviation of each feature and print a report.

        Args:
            ntype (str): the node type of the graph where the features are stored, e.g.
                `atom` and `bond`.

        Returns:
            list: indices of features whose stdev is smaller than the threshold.
        """

        data = self._stack_feature(ntype)
        self.stdevs = np.std(data, axis=0)
        self.means = np.mean(data, axis=0)

        if np.all(self.stdevs <= threshold):
            raise ValueError(
                "No feature meets the stdev threshold {:.5f}".format(threshold)
            )

        feature_name = self.dataset.feature_name[ntype]
        max_name_len = max([len(name) for name in feature_name])
        print("=" * 80)
        print("Node type:", ntype)
        print(
            "feature"
            + " " * max_name_len
            + "stdev    mean   less than threshlod ({})".format(threshold)
        )
        indices = []
        for i, (name, s, m) in enumerate(zip(feature_name, self.stdevs, self.means)):
            if s <= threshold:
                less_than = "yes"
                indices.append(i)
            else:
                less_than = "no"
            fmt = "{:2d} ({})" + " " * (max_name_len - len(name)) + " {:.5f}  {:.5f}  {}"
            print(fmt.format(i, name, s, m, less_than))
        print("=" * 80)

        return indices


class PearsonCorrelation(FeatureAggregator):
    """
    Analyzer to check the Pearson correlation coefficient between the features.
    Args:
        dataset: a :class:`gnn.data.dataset.ElectrolyteData` dataset.
    """

    def compute(self, ntype, exclude=None):
        """
        Compute the Pearson correlation coefficient.

        Args:
            ntype (str): the node type of the graph where the features are stored, e.g.
                `atom` and `bond`.
            exclude (list, optional): indices of features to ignore. This is useful to
                exclude features with 0 stdandard deviation. If `None`, nothing to
                exclude. Defaults to None.
        Returns:
            2D array: correlation between features
        """
        data = self._stack_feature(ntype)
        if exclude is not None:
            data = np.delete(data, exclude, axis=1)
        corr = np.corrcoef(data, rowvar=False)
        return corr


class KMeansAnalyzer(FeatureAggregator):
    """
    KMeans analysis to cluster the features (bond + mean(atom) + max(atom)).

    This only works for the electrolyte dataset.
    """

    def compute(self):
        features, labels = self._stack_bond_feature_plus_atom_feature_and_label()
        return self.embedding(features, labels)

    @staticmethod
    def embedding(features, labels, text_filename="kmeans_electrolyte.txt"):
        """
        Args:
            features (2D array)
            labels: (1D array)
        """
        model = KMeans(n_clusters=10, random_state=35)
        kmeans = model.fit(features)
        clusters = kmeans.predict(features)
        centers = kmeans.cluster_centers_

        return features, clusters, centers, labels


def plot_scatter(features, labels, filename):
    """
    Scatter plot for features and use labels as color.

    Args:
        features (list of 2D array)
        labels (list of 1D array)
        filename (str)
    """
    # plot
    all_marker = ["o", "D", "x", "<", "+"]

    # x and y range
    xmin = min([min(d[:, 0]) for d in features])
    xmax = max([max(d[:, 0]) for d in features])
    ymin = min([min(d[:, 1]) for d in features])
    ymax = max([max(d[:, 1]) for d in features])
    del_x = 0.1 * (xmax - xmin)
    del_y = 0.1 * (ymax - ymin)
    xmin -= del_x
    xmax += del_x
    ymin -= del_y
    ymax += del_y

    # ensure different class has the same color range
    cmin = min([min(i) for i in labels])
    cmax = max([max(i) for i in labels])

    for i, (data, color) in enumerate(zip(features, labels)):
        fig, ax = plt.subplots()
        print("Feature array {} num data points {}".format(i, len(data)))
        X = data[:, 0]
        Y = data[:, 1]
        sc = ax.scatter(
            X,
            Y,
            marker=all_marker[i],
            c=color,
            vmin=cmin,
            vmax=cmax,
            cmap=mpl.cm.viridis,
            ec=None,
        )
        fig.colorbar(sc)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        filename = expand_path(filename)
        fm = os.path.splitext(filename)
        fm = fm[0] + "_" + str(i) + fm[1]
        fig.savefig(fm, bbox_inches="tight")


def plot_heat_map(matrix, labels, filename="heat_map.pdf", cmap=mpl.cm.viridis):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(labels)), minor=False)
    ax.set_yticks(np.arange(len(labels)), minor=False)
    # label them with the respective list entries
    ax.set_xticklabels(labels, minor=False)
    ax.set_yticklabels(labels, minor=False)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # colorbar
    plt.colorbar(im)

    fig.savefig(filename, bbox_inches="tight")
