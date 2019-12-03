import torch
import dgl
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
            if len(samples) == 1:
                graph, label = samples[0]
                return graph, label
            graphs, labels = map(list, zip(*samples))
            batched_graph = dgl.batch_hetero(graphs)
            energies = torch.cat([la["energies"] for la in labels])
            indicators = torch.cat([la["indicators"] for la in labels])
            labels = {"energies": energies, "indicators": indicators}
            return batched_graph, labels

        super(DataLoader, self).__init__(
            dataset, batch_size, shuffle, collate_fn=collate, **kwargs
        )
