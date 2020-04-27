"""Classes and functions for batching multiple graphs together.

The functions defined here are similar to these defined in dgl.batched_graph.py,
but here they are adapted to work with heterograph.
"""

import numpy as np
import torch.nn as nn

from dgl import backend as F
from dgl import BatchedDGLHeteroGraph


READOUT_ON_ATTRS = {
    "nodes": ("batch_num_nodes", "number_of_nodes"),
    "edges": ("batch_num_edges", "number_of_edges"),
}


def _sum_on(graph, typestr, feat, weight):
    """

    Args:
        graph:
        typestr (tuple): with first element of `nodes` or `edges` and the second
            element of the type of nodes or edges.
        feat (str): node data
        weight:

    Returns:

    """
    batch_num_objs_attr, _ = READOUT_ON_ATTRS[typestr[0]]
    data = getattr(getattr(graph, typestr[0])[typestr[1]], "data")
    feat = data[feat]

    if weight is not None:
        weight = data[weight]
        weight = F.reshape(weight, (-1,) + (1,) * (F.ndim(feat) - 1))
        feat = weight * feat

    if isinstance(graph, BatchedDGLHeteroGraph):
        n_graphs = graph.batch_size
        batch_num_objs = getattr(graph, batch_num_objs_attr)(typestr[1])
        seg_id = F.zerocopy_from_numpy(
            np.arange(n_graphs, dtype="int64").repeat(batch_num_objs)
        )
        seg_id = F.copy_to(seg_id, F.context(feat))
        y = F.unsorted_1d_segment_sum(feat, seg_id, n_graphs, 0)
        return y
    else:
        return F.sum(feat, 0)


def _softmax_on(graph, typestr, feat):

    batch_num_objs_attr, _ = READOUT_ON_ATTRS[typestr[0]]
    data = getattr(getattr(graph, typestr[0])[typestr[1]], "data")
    feat = data[feat]

    # TODO: the current solution pads the different graph sizes to the same,
    #  a more efficient way is to use segment sum/max, we need to implement
    #  it in the future.
    if isinstance(graph, BatchedDGLHeteroGraph):
        batch_num_objs = getattr(graph, batch_num_objs_attr)(typestr[1])
        feat = F.pad_packed_tensor(feat, batch_num_objs, -float("inf"))
        feat = F.softmax(feat, 1)
        return F.pack_padded_tensor(feat, batch_num_objs)
    else:
        return F.softmax(feat, 0)


def _broadcast_on(graph, typestr, feat_data):

    batch_num_objs_attr, num_objs_attr = READOUT_ON_ATTRS[typestr[0]]

    if isinstance(graph, BatchedDGLHeteroGraph):
        batch_num_objs = getattr(graph, batch_num_objs_attr)(typestr[1])
        index = []
        for i, num_obj in enumerate(batch_num_objs):
            index.extend([i] * num_obj)
        ctx = F.context(feat_data)
        index = F.copy_to(F.tensor(index), ctx)
        return F.gather_row(feat_data, index)
    else:
        num_objs = getattr(graph, num_objs_attr)()
        if F.ndim(feat_data) == 1:
            feat_data = F.unsqueeze(feat_data, 0)
        return F.cat([feat_data] * num_objs, 0)


def sum_nodes(graph, ntype, feat, weight=None):
    """
    Args:
        graph: graph
        ntype (str): node type
        feat (str): data dict key, the value of which to perform sum.
    Returns:
        (torch.tensor) summed node feature for each graph.
    """
    return _sum_on(graph, ("nodes", ntype), feat, weight)


def softmax_nodes(graph, ntype, feat):
    """
    Args:
        graph: graph
        ntype (str): node type
        feat (str): data dict key, the value of which to perform sum.
    Returns:
        (torch.tensor) softmax node feature for each graph.
    """
    return _softmax_on(graph, ("nodes", ntype), feat)


def broadcast_nodes(graph, ntype, feat_data):
    """
    Args:
        graph: graph
        ntype (str): node type
        feat_data (torch.tensor): data to broadcast to each node of a graph. If a
            batched graph, its size should be equal to the batch size.
    Returns:
        (torch.tensor) tensor
    """

    return _broadcast_on(graph, ("nodes", ntype), feat_data)


class UnifySize(nn.Module):
    """
    A layer to unify the feature size of nodes of different types.
    Each feature uses a linear fc layer to map the size.

    NOTE, after this transformation, each data point is just a linear combination of its
    feature in the original feature space (x_new_ij = x_ik w_kj), there is not mixing of
    feature between data points.

    Args:
        input_dim (dict): feature sizes of nodes with node type as key and size as value
        output_dim (int): output feature size, i.e. the size we will turn all the
            features to
    """

    def __init__(self, input_dim, output_dim):
        super(UnifySize, self).__init__()

        self.linears = nn.ModuleDict(
            {k: nn.Linear(size, output_dim, bias=False) for k, size in input_dim.items()}
        )

    def forward(self, feats):
        """
        Args:
            feats (dict): features dict with node type as key and feature as value

        Returns:
            dict: size adjusted features
        """
        return {k: self.linears[k](x) for k, x in feats.items()}


class LinearN(nn.Module):
    """
    N stacked linear layers.

    Args:
        in_size (int): input feature size
        out_sizes (list): size of each layer
        activations (list): activation function of each layer
        use_bias (list): whether to use bias for the linear layer
    """

    def __init__(self, in_size, out_sizes, activations, use_bias):
        super(LinearN, self).__init__()

        self.fc_layers = nn.ModuleList()
        for out, act, b in zip(out_sizes, activations, use_bias):
            self.fc_layers.append(nn.Linear(in_size, out, bias=b))
            self.fc_layers.append(act)
            in_size = out

    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)
        return x
