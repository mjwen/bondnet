import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class BaseAnalyzer:
    def __init__(self, dataset):
        self.dataset = dataset

    def _stack_feature(self, ntype):
        feature = []
        for g, _ in self.dataset:
            feature.append(g.nodes[ntype].data["feat"])
        return np.concatenate(feature)


class StdevThreshold(BaseAnalyzer):
    """
    Analyzer to check the standard deviation of the features.

    Args:
        dataset: a :class:`gnn.data.dataset.ElectrolyteData` dataset.
    """

    def __init__(self, dataset, threshold=0.0):
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

        print("=" * 80)
        print("Node type:", ntype)
        print("feature  stdev      mean   less than threshod({})".format(self.threshold))
        indices = []
        for i, (s, m) in enumerate(zip(self.stdevs, self.means)):
            if s <= self.threshold:
                less_than = "yes"
                indices.append(i)
            else:
                less_than = "no"
            print("{:4d}    {:.5f}   {:.5f}   {}".format(i, s, m, less_than))
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


def plot_heat_map(matrix, filename="heat_map.pdf", cmap=mpl.cm.viridis):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap=cmap)
    plt.colorbar(im)
    fig.savefig(filename, bbox_inches="tight")
