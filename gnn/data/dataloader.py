import torch
import dgl

# pylint: disable=no-member


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):
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

        super(DataLoader, self).__init__(dataset, collate_fn=collate, **kwargs)


# TODO, could be better to let property be a argument of QM9 dataset, not dataloader
class DataLoaderQM9(torch.utils.data.DataLoader):
    def __init__(self, dataset, property, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'gnn.data', you need not to "
                "provide one"
            )

        supported_properties = [
            "A",
            "B",
            "C",
            "mu",
            "alpha",
            "homo",
            "lumo",
            "gap",
            "r2",
            "zpve",
            "u0",
            "u298",
            "h298",
            "g298",
            "cv",
            "u0_atom",
            "u298_atom",
            "h298_atom",
            "g298_atom",
        ]
        if property not in supported_properties:
            raise ValueError(
                "Property '{}' not supported. Supported ones are: {}".format(
                    property, supported_properties
                )
            )
        property_index = torch.tensor(supported_properties.index(property))

        def collate(samples):
            graphs, labels = map(list, zip(*samples))
            batched_graph = dgl.batch_hetero(graphs)
            labels = torch.index_select(torch.stack(labels), dim=1, index=property_index)
            return batched_graph, labels.view(len(samples), -1)

        super(DataLoaderQM9, self).__init__(dataset, collate_fn=collate, **kwargs)
