"""
Pooling layers
"""
import torch
from torch import nn
from dgl import function as fn


class ConcatenatePooling(nn.Module):
    """
    Concatenate the features of some nodes to other nodes as specified in `etypes`.

    Args:
        etypes (list of tuples): canonical edge types of a graph of which the features
        of node `u` will be concatenated to the features of node `v`.
        For example: if `etypes = [('atom', 'a2b', 'bond'), ('global','g2b', 'bond')]`,
        then the features of `atom` and `global` are concatenated to the features of
        `bond`.
    """

    def __init__(self, etypes):
        super(ConcatenatePooling, self).__init__()
        self.etypes = etypes

    def forward(self, graph, feats):
        graph = graph.local_var()

        # assign data
        for nt, ft in feats.items():
            graph.nodes[nt].data.update({"ft": ft})

        for et in self.etypes:
            graph[et].update_all(
                fn.copy_u("ft", "m"), self._concatenate_reduce_fn, etype=et
            )

        return {nt: graph.nodes[nt].data["ft"] for nt in feats}

    @staticmethod
    def _concatenate_reduce_fn(nodes):
        message = nodes.mailbox["m"]
        # TODO this may have some problem, because it is not permutation independent
        message = message.view(message.shape[0], -1)  # flatten 2nd and 3rd dim
        data = nodes.data["ft"]
        message_shape = message.shape
        data_shape = data.shape
        concatenated = torch.cat((data, message), dim=1)
        return {"ft": concatenated}

