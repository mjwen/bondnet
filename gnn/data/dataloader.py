import torch
import dgl


class DataLoaderBond(torch.utils.data.DataLoader):
    def __init__(self, dataset, hetero=True, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'gnn.data', you need not to "
                "provide one"
            )

        def collate(samples):
            # if len(samples) == 1:
            #     graph, label, scale = samples[0]
            #     return graph, label, scale
            graphs, labels, scales = map(list, zip(*samples))
            if hetero:
                batched_graph = dgl.batch_hetero(graphs)
            else:
                batched_graph = dgl.batch(graphs)
            energies = torch.cat([la["value"] for la in labels])
            indicators = torch.cat([la["indicator"] for la in labels])
            mol_sources = [la["mol_source"] for la in labels]
            # length of value (indicator) for each datapoint i.e. number of bonds in
            # the moleucle from which the bond come from
            length = [len(la["value"]) for la in labels]
            labels = {
                "value": energies,
                "indicator": indicators,
                "mol_source": mol_sources,
                "length": length,
            }
            scales = None if scales[0] is None else torch.cat(scales)
            return batched_graph, labels, scales

        super(DataLoaderBond, self).__init__(dataset, collate_fn=collate, **kwargs)


class DataLoaderMolecule(torch.utils.data.DataLoader):
    def __init__(self, dataset, hetero=True, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'gnn.data', you need not to "
                "provide one"
            )

        def collate(samples):
            graphs, labels, scales = map(list, zip(*samples))
            if hetero:
                batched_graph = dgl.batch_hetero(graphs)
            else:
                batched_graph = dgl.batch(graphs)
            labels = torch.stack(labels)
            scales = None if scales[0] is None else torch.stack(scales)
            return batched_graph, labels, scales

        super(DataLoaderMolecule, self).__init__(dataset, collate_fn=collate, **kwargs)


class DataLoaderBondClassification(torch.utils.data.DataLoader):
    def __init__(self, dataset, hetero=True, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'gnn.data', you need not to "
                "provide one"
            )

        def collate(samples):
            # if len(samples) == 1:
            #     graph, label, scale = samples[0]
            #     return graph, label, scale
            graphs, labels, scales = map(list, zip(*samples))
            if hetero:
                batched_graph = dgl.batch_hetero(graphs)
            else:
                batched_graph = dgl.batch(graphs)
            bond_class = torch.stack([la["class"] for la in labels])
            indicators = [la["indicator"] for la in labels]
            mol_sources = [la["mol_source"] for la in labels]
            labels = {
                "class": bond_class,
                "indicator": indicators,
                "mol_source": mol_sources,
            }
            return batched_graph, labels

        super(DataLoaderBondClassification, self).__init__(
            dataset, collate_fn=collate, **kwargs
        )
