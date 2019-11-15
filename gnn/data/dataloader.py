import torch
import dgl
import copy
from collections import defaultdict

# pylint: disable=no-member


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'gnn.data', you need not to "
                "provide one"
            )

        def collate(samples):
            graphs, labels = map(list, zip(*samples))
            batched_graph = graph_list_to_batch(graphs)
            energies = torch.cat([la["energies"] for la in labels])
            indicators = torch.cat([la["indicators"] for la in labels])
            labels = {"energies": energies, "indicators": indicators}
            return batched_graph, labels

        super(DataLoader, self).__init__(
            dataset, batch_size, shuffle, collate_fn=collate, **kwargs
        )


def graph_list_to_batch(graph_list):

    # query graph info
    g = graph_list[0]
    ntypes = g.ntypes
    etypes = g.canonical_etypes
    attrs = {t: g.nodes[t].data.keys() for t in ntypes}

    # graph connectivity
    current_num_nodes = {t: 0 for t in ntypes}
    connectivity = {t: [] for t in etypes}
    for g in graph_list:
        for t in etypes:
            src, edge, dest = t
            conn = []
            for i in range(g.number_of_nodes(src)):
                i_prime = i + current_num_nodes[src]
                conn.extend(
                    [
                        (i_prime, j + current_num_nodes[dest])
                        for j in g.successors(i, edge)
                    ]
                )
            connectivity[t].extend(conn)
        for t in ntypes:
            current_num_nodes[t] += g.number_of_nodes(t)

    # create batched graph
    batch_g = dgl.heterograph(connectivity)

    # graph data (node only)
    slices = {n: [] for n in ntypes}
    data = {n: defaultdict(list) for n in ntypes}
    for g in graph_list:
        for t in ntypes:
            for a in attrs[t]:
                data[t][a].append(g.nodes[t].data[a])
            slices[t].append(g.number_of_nodes(t))
    # batch data
    for t, dt in data.items():
        for a, d in dt.items():
            data[t][a] = torch.cat(d)
    # add batch data to batch graph
    for t in ntypes:
        batch_g.nodes[t].data.update(data[t])

    # attach graph list and data slices for later split
    batch_g.graph_list = graph_list
    batch_g.node_slices = slices

    return batch_g


def batch_to_graph_list(batch_g):
    ntypes = batch_g.ntypes
    attrs = {t: batch_g.nodes[t].data.keys() for t in ntypes}
    graphs = batch_g.graph_list

    for t in ntypes:
        for a in attrs[t]:
            data = torch.split(batch_g.nodes[t].data[a], batch_g.node_slices[t])
            for g, d in zip(graphs, data):
                g.nodes[t].data.update({a: d})

    return graphs

