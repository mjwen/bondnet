from collections import defaultdict
import numpy as np
import warnings
import torch
from sklearn.preprocessing import StandardScaler as sk_StandardScaler
from dgl import DGLGraph
from gnn.utils import warn_stdout


def _transform(X, copy, with_mean=True, with_std=True, threshold=1.0e-3):
    """
    Args:
        X: a list of 1D tensor or a 2D tensor
    Returns:
        rst: 2D array
        mean: 1D array
        std: 1D array
    """
    if isinstance(X, list):
        X = torch.stack(X)
    scaler = sk_StandardScaler(copy, with_mean, with_std)
    rst = scaler.fit_transform(X)
    mean = scaler.mean_
    std = np.sqrt(scaler.var_)
    for i, v in enumerate(std):
        if v <= threshold:
            warnings.showwarning = warn_stdout
            warnings.warn(
                "Standard deviation for feature {} is {}; smaller than {}. "
                "Take a look at the log file.".format(i, v, threshold)
            )

    return rst, mean, std


class StandardScaler:
    """
    Standardize features using `sklearn.preprocessing.StandardScaler`.

    Args:
        X (2D array): input array

    Returns:
        2D array with each column standardized.
    """

    def __init__(self, copy=True, mean=None, std=None):
        self.copy = copy
        self._mean = mean
        self._std = std

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def __call__(self, X):

        if self._mean is not None and self._std is not None:
            X = (X - self._mean) / self._std
        else:
            X, self._mean, self._std = _transform(X, self.copy)

        return X


class GraphFeatureStandardScaler:
    """
    Standardize features using `sklearn.preprocessing.StandardScaler`.

    Args:
        mean (dict or None):
        std (dict or None):
        
    Returns:
        A list of graphs with updated feats. Note, the input graphs' features are also
        updated in-place.
    """

    def __init__(self, copy=True, mean=None, std=None):
        self.copy = copy
        self._mean = mean
        self._std = std

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

        dtype = node_feats[0].dtype

        # standardize
        if self._mean is not None and self._std is not None:
            node_feats = (torch.cat(node_feats) - self._mean["node"]) / self._std["node"]
            edge_feats = (torch.cat(edge_feats) - self._mean["edge"]) / self._std["edge"]
        else:
            self._std = {}
            self._mean = {}
            node_feats, mean, std = _transform(torch.cat(node_feats), self.copy)
            node_feats = torch.tensor(node_feats, dtype=dtype)
            mean = torch.tensor(mean, dtype=dtype)
            std = torch.tensor(std, dtype=dtype)
            self._mean["node"] = mean
            self._std["node"] = std

            edge_feats, mean, std = _transform(torch.cat(edge_feats), self.copy)
            edge_feats = torch.tensor(edge_feats, dtype=dtype)
            mean = torch.tensor(mean, dtype=dtype)
            std = torch.tensor(std, dtype=dtype)
            self._mean["edge"] = mean
            self._std["edge"] = std

        # assign data back
        node_feats = torch.split(node_feats, node_feats_size)
        edge_feats = torch.split(edge_feats, edge_feats_size)
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

        dtype = node_feats[node_types[0]][0].dtype

        # standardize
        if self._mean is not None and self._std is not None:
            for nt in node_types:
                feats = (torch.cat(node_feats[nt]) - self._mean[nt]) / self._std[nt]
                node_feats[nt] = feats

        else:
            self._std = {}
            self._mean = {}

            for nt in node_types:
                feats, mean, std = _transform(torch.cat(node_feats[nt]), self.copy)
                node_feats[nt] = torch.tensor(feats, dtype=dtype)
                mean = torch.tensor(mean, dtype=dtype)
                std = torch.tensor(std, dtype=dtype)
                self._mean[nt] = mean
                self._std[nt] = std

        # assign data back
        for nt in node_types:
            feats = torch.split(node_feats[nt], node_feats_size[nt])
            for g, ft in zip(graphs, feats):
                g.nodes[nt].data["feat"] = ft

        return graphs
