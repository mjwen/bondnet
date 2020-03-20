import numpy as np
import torch
import dgl
import itertools
from gnn.model.hgat_mol import HGATMol


class HGATReactionNetwork(HGATMol):
    """
    Heterograph attention network for reaction.
    """

    def forward(self, graph, feats, reactions):
        """
        Args:
            graph (DGLHeteroGraph or BatchedDGLHeteroGraph): (batched) molecule graphs
            feats (dict): node features with node type as key and the corresponding
                features as value.
            reactions (list): a sequence of :class:`gnn.data.reaction_network.Reaction`,
                each representing a reaction.

        Returns:
            2D tensor: of shape(N, num_classes), where `num_classes = outdim`.
        """

        # hgat layer
        for layer in self.gat_layers:
            feats = layer(graph, feats)

        # convert mol graphs to reaction graphs by subtracting reactant feats from
        # products feats
        graph, feats = mol_graph_to_rxn_graph(graph, feats, reactions)

        # set2set aggregation
        feats = self.readout_layer(graph, feats)

        # fc, activation, dropout, batch norm
        for layer in self.fc_layers:
            feats = layer(feats)

        return feats


def mol_graph_to_rxn_graph(graph, feats, reactions):
    """
    Convert a batched molecule graph to a batched reaction graph.

    Essentially, a reaction graph has the same graph structure as the reactant and
    its features are the difference between the products features and reactant features.

    Args:
        graph (BatchedDGLHeteroGraph): batched graph representing molecules.
        feats (dict): node features with node type as key and the corresponding
            features as value.
        reactions (list): a sequence of :class:`gnn.data.reaction_network.Reaction`,
            each representing a reaction.

    Returns:
        batched_graph (BatchedDGLHeteroGraph): a batched graph representing a set of
            reactions.
        feats (dict): features for the batched graph
    """
    # should not use graph.local_var() to make a local copy, since it converts a
    # BatchedDGLHeteroGraph into a DGLHeteroGraph. Then unbatch_hetero(graph) below
    # will not work.
    # If you really want to, use copy.deepcopy() to make a local copy

    # assign feats
    for nt, ft in feats.items():
        graph.nodes[nt].data.update({"ft": ft})

    # unbatch molecule graph
    graphs = dgl.unbatch_hetero(graph)
    graphs = np.asarray(graphs)

    # create reaction graphs
    reaction_graphs = []
    reaction_feats = []
    for rxn in reactions:
        reactants = graphs[rxn.reactants]
        products = graphs[rxn.products]

        # whether a molecule has bonds?
        has_bonds = {
            # we support only one reactant now, so no it is assumed always to have bond
            "reactants": [True for _ in reactants],
            "products": [True if len(mp) > 0 else False for mp in rxn.bond_mapping],
        }
        mappings = {"atom": rxn.atom_mapping_as_list, "bond": rxn.bond_mapping_as_list}

        g, fts = create_rxn_graph(reactants, products, mappings, has_bonds)
        reaction_graphs.append(g)
        reaction_feats.append(fts)

    # batched reaction graph and data
    batched_graph = dgl.batch_hetero(reaction_graphs)
    batched_feats = {}
    for nt in feats:
        batched_feats[nt] = torch.cat([ft[nt] for ft in reaction_feats])

    return batched_graph, batched_feats


def create_rxn_graph(reactants, products, mappings, has_bonds, ft_name="ft"):
    """
    A reaction is represented by:

    feats of products - feats of reactant

    Args:
        reactants (list of DGLHeteroGraph): a sequence of reactants graphs
        products (list of DGLHeteroGraph): a sequence of product graphs
        mappings (dict): with node type as the key (e.g. `atom` and `bond`) and a list
            as value, which is a mapping between reactant feature and product feature
            of the same atom (bond).
        has_bonds (dict): whether the reactants and products have bonds.
        ft_name (str): key of feature inf data dict

    Returns:
        graph (DGLHeteroGraph): a reaction graph with feats constructed from between
            reactant and products.
        feats (dict): features of reaction graph
    """
    assert len(reactants) == 1, f"number of reactants ({len(reactants)}) not supported"

    # note, this assumes we have one reactant
    graph = reactants[0]

    feats = dict()
    for nt in graph.ntypes:
        reactants_ft = [p.nodes[nt].data[ft_name] for p in reactants]
        products_ft = [p.nodes[nt].data[ft_name] for p in products]

        # remove bond ft if the corresponding molecule has no bond
        # this is necessary because, to make heterogeneous graph work, we create
        # factitious bond features for molecule without any bond (i.e. single atom
        # molecule, e.g. H+)
        if nt == "bond":
            # select the ones that has bond
            # note, this assumes only one bond missing in products
            ## reactants_ft = list(itertools.compress(reactants_ft, has_bonds["reactants"]))
            products_ft = list(itertools.compress(products_ft, has_bonds["products"]))

            # add a feature with all zeros for the broken bond
            products_ft.append(torch.zeros(1, reactants_ft[0].shape[1]))

        reactants_ft = torch.cat(reactants_ft)
        products_ft = torch.cat(products_ft)

        if nt == "global":
            reactants_ft = torch.sum(reactants_ft, dim=0, keepdim=True)
            products_ft = torch.sum(products_ft, dim=0, keepdim=True)
        else:
            # reorder products_ft such that atoms (bonds) have the same order as reactants
            assert len(products_ft) == len(mappings[nt]), (
                f"products_ft ({len(products_ft)}) and mappings[{nt}] "
                f"({len(mappings[nt])}) have different length"
            )
            products_ft = products_ft[mappings[nt]]

        feats[nt] = products_ft - reactants_ft

    return graph, feats
