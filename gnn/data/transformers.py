from collections import defaultdict
import numpy as np
import warnings
import torch
from sklearn.preprocessing import StandardScaler as sk_StandardScaler
from dgl import DGLGraph


def _transform(X, copy, with_mean, with_std, threshold=1.0e-3):
    """
    Args:
        X: a list of 1D tensor or a 2D tensor
    Returns:
        rst: 2D tensor
        mean: 1D tensor
        std: 1D tensor
    """
    if isinstance(X, list):
        dtype = X[0].dtype
        X = torch.stack(X)
    else:
        dtype = X.dtype
    scaler = sk_StandardScaler(copy, with_mean, with_std)
    rst = scaler.fit_transform(X)
    rst = torch.as_tensor(rst, dtype=dtype)
    mean = torch.as_tensor(scaler.mean_, dtype=dtype)
    std = torch.as_tensor(np.sqrt(scaler.var_), dtype=dtype)
    for i, v in enumerate(std):
        if v <= threshold:
            warnings.warn(
                "Standard deviation for feature {} is {}; smaller than {}. "
                "Take a look at the log file.".format(i, v, threshold)
            )

    return rst, mean, std


class StandardScaler:
    """
    Standardize features using `sklearn.preprocessing.StandardScaler`.

    Args:
        X (2D tensor): input array

    Returns:
        2D tensor with each column standardized.
    """

    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self._mean = None
        self._std = None

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def __call__(self, X):
        X, self._mean, self._std = _transform(X, self.copy, self.with_mean, self.with_std)
        return X


class GraphFeatureStandardScaler:
    """
    Standardize features using `sklearn.preprocessing.StandardScaler`.

    Args:
        graphs (list of graphs): the list of graphs where the features are stored.

    Returns:
        A list of graphs with updated feats. Note, the input graphs' features are also
        updated in-place.
    """

    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self._mean = None
        self._std = None

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def __call__(self, graphs):
        g = graphs[0]
        if isinstance(g, DGLGraph):
            return self._homo_graph(graphs)
        else:
            return self._hetero_graph(graphs)

    def _homo_graph(self, graphs):
        node_feats = []
        node_feats_size = []
        edge_feats = []
        edge_feats_size = []

        # obtain feats from graphs
        for g in graphs:
            data = g.ndata["feat"]
            node_feats.append(data)
            node_feats_size.append(len(data))

            data = g.edata["feat"]
            edge_feats.append(data)
            edge_feats_size.append(len(data))

        # standardize
        self._std = {}
        self._mean = {}
        node_feats, mean, std = _transform(
            torch.cat(node_feats), self.copy, self.with_mean, self.with_std
        )
        self._mean["node"] = mean
        self.std["node"] = std
        edge_feats, mean, std = _transform(
            torch.cat(edge_feats), self.copy, self.with_mean, self.with_std
        )
        self._mean["edge"] = mean
        self._std["edge"] = std
        node_feats = torch.split(node_feats, node_feats_size)
        edge_feats = torch.split(edge_feats, edge_feats_size)

        # assign data back
        for g, n, e in zip(graphs, node_feats, edge_feats):
            g.ndata["feat"] = n
            g.edata["feat"] = e
        return graphs

    def _hetero_graph(self, graphs):
        g = graphs[0]
        node_types = g.ntypes
        node_feats = defaultdict(list)
        node_feats_size = defaultdict(list)

        # obtain feats from graphs
        for g in graphs:
            for nt in node_types:
                data = g.nodes[nt].data["feat"]
                node_feats[nt].append(data)
                node_feats_size[nt].append(len(data))

        # standardize and update
        self._std = {}
        self._mean = {}
        for nt in node_feats:
            feats, mean, std = _transform(
                torch.cat(node_feats[nt]), self.copy, self.with_mean, self.with_std
            )
            self._mean[nt] = mean
            self._std[nt] = std
            feats = torch.split(feats, node_feats_size[nt])
            for g, ft in zip(graphs, feats):
                g.nodes[nt].data["feat"] = ft
        return graphs
