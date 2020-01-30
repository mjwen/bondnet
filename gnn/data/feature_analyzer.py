import os
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib as mpl
import matplotlib.pyplot as plt
from gnn.layer.readout import ConcatenateMeanMax
from gnn.utils import expand_path


def get_id(s):
    """
    Get the id from a string `aaa_bbb_id_ccc`.

    Returns:
        str: id
    """
    return s.split("_")[2]


def read_sdf(filename):
    """
    Read sdf file.

    Returns:
        dict: with mol id as key and the sdf struct body as val.
    """
    structs = dict()

    filename = expand_path(filename)
    with open(filename, "r") as f:
        for line in f:
            if "int_id" in line:
                key = get_id(line.split()[0])
                body = line
            elif "$$$$" in line:
                structs[key] = body
            else:
                body += line

    return structs


def read_label(filename, sort_by_formula=True):
    """
    Read label file.

    Args:
        sort_by_formula (bool): sort the returned list by formula or not.

    Returns:
        list of dict: dict has keys `raw`, `formula`, `reactants`, `products`, and the
        keys corresponds to they are str, str, str, and list of str.

        for reactants and products, mol id will be the values.
    """
    labels = []

    filename = expand_path(filename)
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue

            pattern = "\[(.*?)\]"
            result = re.findall(pattern, line)

            tmp = result[0].strip(" '")
            reactants = get_id(tmp)
            formula = tmp.split("_")[0]

            products = result[1]
            if "," in products:  # two or more products
                products = products.split(",")
            else:  # one products
                products = [products]
            products = [get_id(p.strip(" '")) for p in products]

            d = {
                "raw": line,
                "reactants": reactants,
                "products": products,
                "formula": formula,
            }
            labels.append(d)

        # sort by formula
        if sort_by_formula:
            labels = sorted(labels, key=lambda d: d["formula"])

    return labels


class BaseAnalyzer:
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


class StdevThreshold(BaseAnalyzer):
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


class PearsonCorrelation(BaseAnalyzer):
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


class KMeansAnalyzer(BaseAnalyzer):
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


class PCAAnalyzer(BaseAnalyzer):
    """
    PCA analysis of the bond descriptor and bond energy.

    This only works for the electrolyte dataset.
    """

    def compute(self):
        # features, labels = self._stack_feature_and_label(ntype="bond")
        features, labels = self._stack_bond_feature_plus_atom_feature_and_label()
        return self.embedding([features], [labels])

    @staticmethod
    def embedding(
        features,
        labels,
        text_filename="PCA_electrolyte.txt",
        plot_filename="PCA_electrolyte.pdf",
    ):
        """
        Args:
            features (list of 2D array): all data will be used for training the model,
                and each array will be evaluated separately.
            labels: (list of 1D array): labels for the features.
        """
        model = PCA(n_components=2)
        embeddings = model.fit_transform(np.concatenate(features))
        sizes = [len(d) for d in features]
        indices = [sum(sizes[:i]) for i in range(1, len(sizes))]
        embeddings = np.split(embeddings, indices)

        write_embeddings(embeddings, labels, text_filename)
        plot_scatter(embeddings, labels, plot_filename)


class TSNEAnalyzer(BaseAnalyzer):
    """
    TSNE analysis of the bond descriptor and bond energy.

    This only works for the electrolyte dataset.
    """

    def compute(self):
        # features, labels = self._stack_feature_and_label(ntype="bond")
        features, labels = self._stack_bond_feature_plus_atom_feature_and_label()
        return self.embedding([features], [labels])

    @staticmethod
    def embedding(
        features,
        labels,
        text_filename="TSNE_electrolyte.txt",
        plot_filename="TSNE_electrolyte.pdf",
    ):
        """
        Args:
            features (list of 2D array): all data will be used for training the model,
                and each array will be evaluated separately.
            labels: (list of 1D array): labels for the features.
        """

        model = TSNE(n_components=2)
        embeddings = model.fit_transform(np.concatenate(features))
        sizes = [len(d) for d in features]
        indices = [sum(sizes[:i]) for i in range(1, len(sizes))]
        embeddings = np.split(embeddings, indices)

        write_embeddings(embeddings, labels, text_filename)
        plot_scatter(embeddings, labels, plot_filename)


def write_embeddings(features, labels, filename):
    with open(filename, "w") as f:
        f.write("# components...   label\n")
        for idx, (emb, lb) in enumerate(zip(features, labels)):
            f.write("\n\n# {}\n".format(idx))
            for i, j in zip(emb, lb):
                for k in i:
                    f.write("{:14.6e}".format(k))
                f.write("   {:14.6e}\n".format(j))


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
