"""
Pooling layers
"""
import torch
from torch import nn
from dgl import function as fn
from dgl import BatchedDGLHeteroGraph
from gnn.layer.utils import broadcast_nodes, sum_nodes, softmax_nodes


class ConcatenateMeanMax(nn.Module):
    """
    Concatenate the features of some nodes to other nodes as specified in `etypes`.

    Args:
        etypes (list of tuples): canonical edge types of a graph of which the features
            of node `u` will be concatenated to the features of node `v`.
            For example: if `etypes = [('atom', 'a2b', 'bond'), ('global','g2b', 'bond')]`
            then the features of `atom` and `global` are concatenated to the features of
            `bond`.
    """

    def __init__(self, etypes):
        super(ConcatenateMeanMax, self).__init__()
        self.etypes = etypes

    def forward(self, graph, feats):
        graph = graph.local_var()

        # assign data
        for nt, ft in feats.items():
            graph.nodes[nt].data.update({"ft": ft})

        for et in self.etypes:
            # graph[et].update_all(fn.copy_u("ft", "m"), fn.mean("m", "mean"), etype=et)
            # graph[et].update_all(fn.copy_u("ft", "m"), fn.max("m", "max"), etype=et)
            # graph.apply_nodes(self._concatenate_node_feat, ntype=et[2])
            graph[et].update_all(
                fn.copy_u("ft", "m"), self._concatenate_mean_max, etype=et
            )

        return {nt: graph.nodes[nt].data["ft"] for nt in feats}

    # @staticmethod
    # def _concatenate_node_feat(nodes):
    #     data = nodes.data["ft"]
    #     mean = nodes.data["mean"]
    #     max = nodes.data["max"]
    #     concatenated = torch.cat((data, mean, max), dim=1)
    #     return {"ft": concatenated}

    @staticmethod
    def _concatenate_mean_max(nodes):
        message = nodes.mailbox["m"]
        mean_v = torch.mean(message, dim=1)
        max_v = torch.max(message, dim=1).values
        data = nodes.data["ft"]
        concatenated = torch.cat((data, mean_v, max_v), dim=1)
        return {"ft": concatenated}


class Set2Set(nn.Module):
    r"""Apply Set2Set (`Order Matters: Sequence to sequence for sets
    <https://arxiv.org/pdf/1511.06391.pdf>`__) over the nodes in the graph.

    For each individual graph in the batch, set2set computes

    .. math::
        q_t &= \mathrm{LSTM} (q^*_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(x_i \cdot q_t)

        r_t &= \sum_{i=1}^N \alpha_{i,t} x_i

        q^*_t &= q_t \Vert r_t

    for this graph.

    Parameters
    ----------
    input_dim : int
        Size of each input sample
    n_iters : int
        Number of iterations.
    n_layers : int
        Number of recurrent layers.
    n_type : str
        Node type
    """

    def __init__(self, input_dim, n_iters, n_layers, ntype):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.ntype = ntype
        self.lstm = torch.nn.LSTM(self.output_dim, self.input_dim, n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        self.lstm.reset_parameters()

    def forward(self, graph, feat):
        r"""Compute set2set pooling.

        Parameters
        ----------
        graph : DGLGraph or BatchedDGLGraph
            The graph.
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)` where
            :math:`N` is the number of nodes in the graph.
        ntype: string
            Node type to apply set2set.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(D)` (if
            input graph is a BatchedDGLGraph, the result shape
            would be :math:`(B, D)`.
        """
        with graph.local_scope():
            batch_size = 1
            if isinstance(graph, BatchedDGLHeteroGraph):
                batch_size = graph.batch_size

            h = (
                feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
                feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
            )

            q_star = feat.new_zeros(batch_size, self.output_dim)

            for _ in range(self.n_iters):
                q, h = self.lstm(q_star.unsqueeze(0), h)
                q = q.view(batch_size, self.input_dim)

                e = (feat * broadcast_nodes(graph, self.ntype, q)).sum(
                    dim=-1, keepdim=True
                )
                graph.nodes[self.ntype].data["e"] = e
                alpha = softmax_nodes(graph, self.ntype, "e")

                graph.nodes[self.ntype].data["r"] = feat * alpha
                readout = sum_nodes(graph, self.ntype, "r")

                if readout.dim() == 1:  # graph is not a BatchedDGLGraph
                    readout = readout.unsqueeze(0)

                q_star = torch.cat([q, readout], dim=-1)

            if isinstance(graph, BatchedDGLHeteroGraph):
                return q_star
            else:
                return q_star.squeeze(0)

    def extra_repr(self):
        """Set the extra representation of the module.
        which will come into effect when printing the model.
        """
        summary = "n_iters={n_iters}"
        return summary.format(**self.__dict__)


class Set2SetThenCat(nn.Module):
    """
    Set2Set for nodes (separate for different node type) and then concatenate to create a
    representation of the graph.

     Args:
        n_iter (int): number of LSTM iteration
        n_layer (int): number of LSTM layers
        ntypes (list of str): node types to perform Set2Set.
        in_feats (list of int): feature size of nodes corresponds to ntypes
        ntypes_direct_cat (list of str): node types on which not perform Set2Set, but
            need to concatenate their feature directly.
    """

    def __init__(self, n_iters, n_layer, ntypes, in_feats, ntypes_direct_cat):
        super(Set2SetThenCat, self).__init__()
        self.ntypes = ntypes
        self.ntypes_direct_cat = ntypes_direct_cat

        self.layers = nn.ModuleDict()
        for nt, sz in zip(ntypes, in_feats):
            self.layers[nt] = Set2Set(
                input_dim=sz, n_iters=n_iters, n_layers=n_layer, ntype=nt
            )

    def forward(self, graph, feats):
        rst = []
        for nt in self.ntypes:
            ft = self.layers[nt](graph, feats[nt])
            rst.append(ft)
        for nt in self.ntypes_direct_cat:
            ft = feats[nt]
            rst.append(ft)
        return torch.cat(rst, dim=-1)  # dim=-1 to deal with batched graph
