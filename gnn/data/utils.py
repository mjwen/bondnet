from collections import defaultdict
from sklearn.preprocessing import StandardScaler as sk_StandardScaler
from dgl import DGLGraph
import numpy as np
import torch
import warnings


class StandardScaler:
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
        node_feats, mean, std = self._transform(node_feats)
        self._mean["node"] = mean
        self.std["node"] = std
        edge_feats, mean, std = self._transform(edge_feats)
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
            feats, mean, std = self._transform(node_feats[nt])
            self._mean[nt] = mean
            self._std[nt] = std
            feats = torch.split(feats, node_feats_size[nt])
            for g, ft in zip(graphs, feats):
                g.nodes[nt].data["feat"] = ft
        return graphs

    def _transform(self, X, threshold=1.0e-3):
        """
        X: a list of values
        """
        dtype = X[0].dtype
        X = np.concatenate(X)
        scaler = sk_StandardScaler(self.copy, self.with_mean, self.with_std)
        rst = scaler.fit_transform(X)
        mean = scaler.mean_
        std = np.sqrt(scaler.var_)
        for i, v in enumerate(std):
            if v <= threshold:
                warnings.warn(
                    "Standard deviation for feature {} is {}; smaller than {}. "
                    "Take a look at the log file.".format(i, v, threshold)
                )

        return torch.as_tensor(rst, dtype=dtype), mean, std


def graph_struct_list_representation(g):
    """
    Represent the heterogrpah canonical edges as a dict.

    Args:
        g: a dgl heterograph

    Returns:
        dict of dict of list: dest node is the outer key; `nodes`and `edges` are the
        innter keys; src nodes are the values associated with `nodes` and edge types
        are the vlues associated with `edges`.

    Example:
        Suppose the graph has canonical edges:
         src   edge   dest

        [['A', 'A2B', 'B'],
         ['C', 'C2B', 'B'],
         ['C', 'C2A', 'A']]

        This function rturns:
        {
         'A':{'nodes':['C'],
              'edges':['C2A']},
         'B':{'nodes':['A', 'C'],
              'edges':['A2B', 'C2B']}
        }
    """
    graph_strcut = {
        t: {"nodes": defaultdict(list), "edges": defaultdict(list)} for t in g.ntypes
    }
    for src, edge, dest in g.canonical_etypes:
        graph_strcut[dest]["nodes"].append(src)
        graph_strcut[dest]["edges"].append(edge)

    return graph_strcut


def get_bond_to_atom_map(g):
    """
    Query which atoms are associated with the bonds.

    Args:
        g: dgl heterograph

    Returns:
        dict: with bond index as the key and a tuple of atom indices of atoms that
            form the bond.
    """
    nbonds = g.number_of_nodes("bond")
    bond_to_atom_map = dict()
    for i in range(nbonds):
        atoms = g.successors(i, "b2a")
        bond_to_atom_map[i] = sorted(atoms)
    return bond_to_atom_map


def get_atom_to_bond_map(g):
    """
    Query which bonds are associated with the atoms.

    Args:
        g: dgl heterograph

    Returns:
        dict: with atom index as the key and a list of indices of bonds is
        connected to the atom.
    """
    natoms = g.number_of_nodes("atom")
    atom_to_bond_map = dict()
    for i in range(natoms):
        bonds = g.successors(i, "a2b")
        atom_to_bond_map[i] = sorted(list(bonds))
    return atom_to_bond_map
