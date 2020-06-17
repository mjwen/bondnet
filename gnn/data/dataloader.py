import numpy as np
import torch
import dgl


class DataLoader(torch.utils.data.DataLoader):
    """
    This dataloader works for the case where the labels of all data points are of the
    same shape. For example, regression on molecule energy.
    """

    def __init__(self, dataset, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'gnn.data', you need not to "
                "provide one"
            )

        def collate(samples):
            graphs, labels = map(list, zip(*samples))

            g = graphs[0]
            if isinstance(g, dgl.DGLGraph):
                batched_graphs = dgl.batch(graphs)
            elif isinstance(g, dgl.DGLHeteroGraph):
                batched_graphs = dgl.batch_hetero(graphs)
            else:
                raise ValueError(
                    f"graph type {g.__class__.__name__} not supported. Should be either "
                    f"dgl.DGLGraph or dgl.DGLHeteroGraph."
                )

            batched_labels = torch.utils.data.dataloader.default_collate(labels)

            return batched_graphs, batched_labels

        super(DataLoader, self).__init__(dataset, collate_fn=collate, **kwargs)


class DataLoaderGraphNorm(torch.utils.data.DataLoader):
    """
    This dataloader works for the case where the label of each data point are of the
    same shape. For example, regression on molecule energy.
    """

    def __init__(self, dataset, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'gnn.data', you need not to "
                "provide one"
            )

        def collate(samples):
            graphs, labels = map(list, zip(*samples))

            g = graphs[0]
            if isinstance(g, dgl.DGLGraph):
                batched_graphs = dgl.batch(graphs)
                sizes_atom = [g.number_of_nodes() for g in graphs]
                sizes_bond = [g.number_of_edges() for g in graphs]

            elif isinstance(g, dgl.DGLHeteroGraph):
                batched_graphs = dgl.batch_hetero(graphs)
                sizes_atom = [g.number_of_nodes("atom") for g in graphs]
                sizes_bond = [g.number_of_nodes("bond") for g in graphs]

            else:
                raise ValueError(
                    f"graph type {g.__class__.__name__} not supported. Should be either "
                    f"dgl.DGLGraph or dgl.DGLHeteroGraph."
                )

            batched_labels = torch.utils.data.dataloader.default_collate(labels)

            norm_atom = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_atom]
            norm_bond = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_bond]
            batched_labels["norm_atom"] = 1.0 / torch.cat(norm_atom).sqrt()
            batched_labels["norm_bond"] = 1.0 / torch.cat(norm_bond).sqrt()

            return batched_graphs, batched_labels

        super(DataLoaderGraphNorm, self).__init__(dataset, collate_fn=collate, **kwargs)


class DataLoaderBond(torch.utils.data.DataLoader):
    """
    This dataloader works for bond related dataset, where bond specific properties (
    e.g. bond energy) needs to be ber specified by an index.
    """

    def __init__(self, dataset, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'gnn.data', you need not to "
                "provide one"
            )

        def collate(samples):
            graphs, labels = map(list, zip(*samples))

            g = graphs[0]
            if isinstance(g, dgl.DGLGraph):
                batched_graphs = dgl.batch(graphs)
            elif isinstance(g, dgl.DGLHeteroGraph):
                batched_graphs = dgl.batch_hetero(graphs)
            else:
                raise ValueError(
                    f"graph type {g.__class__.__name__} not supported. Should be either "
                    f"dgl.DGLGraph or dgl.DGLHeteroGraph."
                )

            value = torch.cat([la["value"] for la in labels])
            # index_0 = labels[0]["bond_index"]
            # indices = [
            #     labels[i]["bond_index"] + labels[i - 1]["num_bonds_in_molecule"]
            #     for i in range(1, len(labels))
            # ]
            # indices = torch.stack(index_0 + indices)
            num_bonds = torch.stack([la["num_bonds_in_molecule"] for la in labels])
            staring_index = torch.cumsum(num_bonds, dim=0)
            index = torch.cat(
                [
                    la["bond_index"]
                    if i == 0
                    else la["bond_index"] + staring_index[i - 1]
                    for i, la in enumerate(labels)
                ]
            )
            batched_labels = {"value": value, "index": index}

            # add label scaler if it is used
            try:
                mean = [la["scaler_mean"] for la in labels]
                stdev = [la["scaler_stdev"] for la in labels]
                batched_labels["scaler_mean"] = torch.cat(mean)
                batched_labels["scaler_stdev"] = torch.cat(stdev)
            except KeyError:
                pass

            return batched_graphs, batched_labels

        super(DataLoaderBond, self).__init__(dataset, collate_fn=collate, **kwargs)


class DataLoaderReaction(torch.utils.data.DataLoader):
    """
    This dataloader works specifically for the reaction dataset where each reaction is
    represented by a list of the molecules (i.e. reactants and products).

    Also, the label value of each datapoint should be of the same shape.
    """

    def __init__(self, dataset, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'gnn.data', you need not to "
                "provide one"
            )

        def collate(samples):
            graphs, labels = map(list, zip(*samples))

            # note each element of graph is a list of mol graphs that constitute a rxn
            graphs = np.concatenate(graphs)
            g = graphs[0]
            if isinstance(g, dgl.DGLGraph):
                batched_graphs = dgl.batch(graphs)
            elif isinstance(g, dgl.DGLHeteroGraph):
                batched_graphs = dgl.batch_hetero(graphs)
            else:
                raise ValueError(
                    f"graph type {g.__class__.__name__} not supported. Should be either "
                    f"dgl.DGLGraph or dgl.DGLHeteroGraph."
                )

            target_class = torch.stack([la["value"] for la in labels])
            atom_mapping = [la["atom_mapping"] for la in labels]
            bond_mapping = [la["bond_mapping"] for la in labels]
            global_mapping = [la["global_mapping"] for la in labels]
            num_mols = [la["num_mols"] for la in labels]
            identifier = [la["id"] for la in labels]

            batched_labels = {
                "value": target_class,
                "atom_mapping": atom_mapping,
                "bond_mapping": bond_mapping,
                "global_mapping": global_mapping,
                "num_mols": num_mols,
                "id": identifier,
            }

            # add label scaler if it is used
            try:
                mean = [la["scaler_mean"] for la in labels]
                stdev = [la["scaler_stdev"] for la in labels]
                batched_labels["scaler_mean"] = torch.stack(mean)
                batched_labels["scaler_stdev"] = torch.stack(stdev)
            except KeyError:
                pass

            return batched_graphs, batched_labels

        super(DataLoaderReaction, self).__init__(dataset, collate_fn=collate, **kwargs)


class DataLoaderReactionNetwork(torch.utils.data.DataLoader):
    """
    This dataloader works specifically for the reaction network where a the reactions
    are constructed from a list of reactions.
    """

    def __init__(self, dataset, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally by 'gnn.data', you need not to "
                "provide one"
            )

        def collate(samples):
            rn, rxn_ids, labels = map(list, zip(*samples))

            # each element of `rn` is the same reaction network
            reactions, graphs = rn[0].subselect_reactions(rxn_ids)

            g = graphs[0]
            if isinstance(g, dgl.DGLGraph):
                batched_graphs = dgl.batch(graphs)
                sizes_atom = [g.number_of_nodes() for g in graphs]
                sizes_bond = [g.number_of_edges() for g in graphs]

            elif isinstance(g, dgl.DGLHeteroGraph):
                batched_graphs = dgl.batch_hetero(graphs)
                sizes_atom = [g.number_of_nodes("atom") for g in graphs]
                sizes_bond = [g.number_of_nodes("bond") for g in graphs]

            else:
                raise ValueError(
                    f"graph type {g.__class__.__name__} not supported. Should be either "
                    f"dgl.DGLGraph or dgl.DGLHeteroGraph."
                )

            target = torch.stack([la["value"] for la in labels])
            identifier = [la["id"] for la in labels]

            batched_labels = {
                "value": target,
                "id": identifier,
                "reaction": reactions,
            }

            # add label scaler if it is used
            try:
                mean = [la["scaler_mean"] for la in labels]
                stdev = [la["scaler_stdev"] for la in labels]
                batched_labels["scaler_mean"] = torch.stack(mean)
                batched_labels["scaler_stdev"] = torch.stack(stdev)
            except KeyError:
                pass

            # graph norm
            norm_atom = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_atom]
            norm_bond = [torch.FloatTensor(s, 1).fill_(s) for s in sizes_bond]
            batched_labels["norm_atom"] = 1.0 / torch.cat(norm_atom).sqrt()
            batched_labels["norm_bond"] = 1.0 / torch.cat(norm_bond).sqrt()

            return batched_graphs, batched_labels

        super(DataLoaderReactionNetwork, self).__init__(
            dataset, collate_fn=collate, **kwargs
        )
