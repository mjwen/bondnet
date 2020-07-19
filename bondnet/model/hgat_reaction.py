"""
Heterogeneous Graph Attention Networks on reaction level property.
"""

import dgl
import torch
from bondnet.model.hgat_mol import HGATMol
from bondnet.utils import np_split_by_size


class HGATReaction(HGATMol):
    """
    Heterograph attention network for reaction.
    """

    def forward(self, graph, feats, num_mols, atom_mapping, bond_mapping, global_mapping):
        """
        Args:
            graph (DGLHeteroGraph or BatchedDGLHeteroGraph): (batched) molecule graphs
            feats (dict): node features with node type as key and the corresponding
                features as value.
            num_mols (list): number of molecules (reactant plus products) in each
                reactions.
            atom_mapping (list of list): each inner list contains the atom mapping
                (a dict) from products to reactant.
            bond_mapping (list of list): each inner list contains the bond mapping
                (a dict) from products to reactant.
            global_mapping (list of list): each inner list contains the mapping (dict) of
                global feat index between products and reactant.

        Returns:
            2D tensor: of shape(N, num_classes), where `num_classes = outdim`.
        """

        # hgat layer
        for i, layer in enumerate(self.gat_layers):
            feats = layer(graph, feats)

            # apply activation after average over heads (eq. 6 of the GAT paper)
            # see below in forward()
            if i == len(self.gat_layers) - 1:
                for nt in feats:
                    ft = feats[nt].view(feats[nt].shape[0], self.num_heads, -1)
                    feats[nt] = self.gat_activation(torch.mean(ft, dim=1))

        # convert mol graphs to reaction graphs, i.e. subtracting reactant feat from
        # products feat
        graph, feats = mol_graph_to_rxn_graph(
            graph, feats, num_mols, atom_mapping, bond_mapping, global_mapping
        )

        # set2set aggregation
        feats = self.readout_layer(graph, feats)

        # fc, activation, dropout, batch norm
        for layer in self.fc_layers:
            feats = layer(feats)

        return feats


def mol_graph_to_rxn_graph(
    graph, feats, num_mols, atom_mapping, bond_mapping, global_mapping
):
    """
    Convert a batched molecule graph to a batched reaction graph.

    Essentially, a reaction graph has the same graph structure as the reactant and
    its features are the difference between the products features and reactant features.

    Args:
        graph (BatchedDGLHeteroGraph): batched graph representing molecules.
        feats (dict): node features with node type as key and the corresponding
            features as value.
        num_mols (list): number of molecules (reactant and products) in the
            reactions.
        atom_mapping (list of list): each inner list contains the mapping (dict) of
            atom index between products and reactant.
        bond_mapping (list of list): each inner list contains the mapping (dict) of
            bond index between products and reactant.
        global_mapping (list of list): each inner list contains the mapping (dict) of
            global feat index between products and reactant.

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

    # unbatch graph
    graphs = dgl.unbatch_hetero(graph)
    reactions = np_split_by_size(graphs, num_mols)
    reactants = [rxn[0] for rxn in reactions]
    products = [rxn[1:] for rxn in reactions]

    # get graph using rxn feats
    graphs = []
    for r, p, amp, bmp, gmp in zip(
        reactants, products, atom_mapping, bond_mapping, global_mapping
    ):
        mappings = {"atom": amp, "bond": bmp, "global": gmp}
        g = create_rxn_graph(r, p, mappings)
        graphs.append(g)

    # batch graph
    batched_graph = dgl.batch_hetero(graphs)
    feats = {nt: batched_graph.nodes[nt].data["ft"] for nt in batched_graph.ntypes}

    return batched_graph, feats


def create_rxn_graph(reactant, products, mappings, ft_name="ft"):
    """
    A reaction is represented by:

    feats of products - feats of reactant

    Args:
        reactant (DGLHeteroGraph): graph of the reactant
        products (list of DGLHeteroGraph): a sequence of product graphs
        mappings (dict): with node type as the key (e.g. `atom`, `bond`, and
            `global`) and a list of mapping (dict) between product feat index
            and reactant feat index.
        ft_name (str): key of feature inf data dict

    Returns:
        DGLHeteroGraph: a reaction graph with feats constructed from between
            reactant and products.
    """
    graph = reactant

    for nt in graph.ntypes:

        # negating reactant feats
        ft = -graph.nodes[nt].data[ft_name]

        # add products feats
        for i, p in enumerate(products):
            mp = mappings[nt][i]
            # product may not have certain type of node (e.g. H does not have `bond`
            # node). In this case, its mapping mp is empty.
            if mp:
                p_ft = p.nodes[nt].data[ft_name]
                for p_idx, r_idx in mp.items():
                    ft[r_idx] += p_ft[p_idx]

        # assign back to graph
        graph.nodes[nt].data.update({ft_name: ft})

    return graph
