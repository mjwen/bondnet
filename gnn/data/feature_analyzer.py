import numpy as np
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt


class BaseAnalyzer:
    def __init__(self, dataset):
        self.dataset = dataset

    def _stack_feature(self, ntype):
        feature = []
        for g, _, _ in self.dataset:
            feature.append(g.nodes[ntype].data["feat"])
        return np.concatenate(feature)


class StdevThreshold(BaseAnalyzer):
    """
    Analyzer to check the standard deviation of the features.

    Args:
        dataset: a :class:`gnn.data.dataset.ElectrolyteData` dataset.
    """

    def __init__(self, dataset, threshold=1e-8):
        super(StdevThreshold, self).__init__(dataset)
        self.threshold = threshold

    def compute(self, ntype):
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

        if np.all(self.stdevs <= self.threshold):
            raise ValueError(
                "No feature meets the stdev threshold {:.5f}".format(self.threshold)
            )

        feature_name = self.dataset.feature_name[ntype]
        max_name_len = max([len(name) for name in feature_name])
        print("=" * 80)
        print("Node type:", ntype)
        print(
            "feature"
            + " " * max_name_len
            + "stdev    mean   less than threshlod ({})".format(self.threshold)
        )
        indices = []
        for i, (name, s, m) in enumerate(zip(feature_name, self.stdevs, self.means)):
            if s <= self.threshold:
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

    def __init__(self, dataset):
        super(PearsonCorrelation, self).__init__(dataset)

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


class PCABondFeature:
    """
    PCA analysis of the bond descriptor and bond energy.

    This only works for the electrolyte dataset.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def compute(self):
        ntype = "bond"

        features = []
        labels = []
        for g, lb, _ in self.dataset:
            # indices of bond that has energy
            indices = [int(i) for i, v in enumerate(lb["indicator"]) if v == 1]

            labels.append(lb["value"][indices])
            features.append(g.nodes[ntype].data["feat"][indices])

        features = np.concatenate(features)
        labels = np.concatenate(labels)

        pca = PCA(n_components=2)
        features = pca.fit_transform(features)

        with open("PCA_electrolyte.txt", "w") as f:
            f.write("# PCA components...   label\n")
            for i, j in zip(features, labels):
                for k in i:
                    f.write("{:14.6e}".format(k))
                f.write("   {:14.6e}\n".format(j))

        self._plot(features, labels)

    @staticmethod
    def _plot(data, color, filename="PCA_electrolyte.pdf"):
        print("Number of data points", len(data))
        X = data[:, 0]
        Y = data[:, 1]
        fig, ax = plt.subplots()
        sc = ax.scatter(X, Y, c=color, cmap=mpl.cm.viridis, ec=None)
        plt.colorbar(sc)
        fig.savefig(filename, bbox_inches="tight")


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
