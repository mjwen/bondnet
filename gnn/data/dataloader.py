import torch
import dgl

# pylint: disable=no-member


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, hetero=True, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'gnn.data', you need not to "
                "provide one"
            )

        def collate(samples):
            if len(samples) == 1:
                graph, label = samples[0]
                return graph, label
            graphs, labels = map(list, zip(*samples))
            if hetero:
                batched_graph = dgl.batch_hetero(graphs)
            else:
                batched_graph = dgl.batch(graphs)
            energies = torch.cat([la["value"] for la in labels])
            indicators = torch.cat([la["indicator"] for la in labels])
            labels = {"value": energies, "indicator": indicators}
            return batched_graph, labels

        super(DataLoader, self).__init__(dataset, collate_fn=collate, **kwargs)


class DataLoaderQM9(torch.utils.data.DataLoader):
    def __init__(self, dataset, hetero=True, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'gnn.data', you need not to "
                "provide one"
            )

        def collate(samples):
            graphs, labels = map(list, zip(*samples))
            if hetero:
                batched_graph = dgl.batch_hetero(graphs)
            else:
                batched_graph = dgl.batch(graphs)
            labels = torch.stack(labels)
            return batched_graph, labels

        super(DataLoaderQM9, self).__init__(dataset, collate_fn=collate, **kwargs)
