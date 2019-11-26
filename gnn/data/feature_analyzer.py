import numpy as np
import warnings
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


class VarianceThreshold(BaseAnalyzer):
    def __init__(self, dataset, threshold=0.0):
        super(VarianceThreshold, self).__init__(dataset)
        self.threshold = threshold

    def compute(self, ntype):
        data = self._stack_feature(ntype)
        self.variances = np.var(data, axis=0)

        if np.all(self.variances <= self.threshold):
            raise ValueError(
                "No feature in X meets the variance threshold {:.5f}".format(
                    self.threshold
                )
            )
        for i, v in enumerate(self.variances):
            if v <= self.threshold:
                warnings.warn(
                    "Feature {} of node type '{}' has a variance <= {}.".format(
                        i, ntype, self.threshold
                    )
                )


class PearsonCorrelation(BaseAnalyzer):
    def __init__(self, dataset):
        super(PearsonCorrelation, self).__init__(dataset)

    def compute(self, ntype, exclude=None):
        """[summary]

        Args:
            ntype (str): node type in the graph
            exclude (list, optional): indices of features to ignore. This is useful to
                exclude features with 0 variance. Defaults to None.
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
    im = ax.imshow(matrix)
    plt.colorbar(im)
    fig.savefig(filename, bbox_inches="tight")
