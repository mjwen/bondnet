"""
Heterogeneous Graph Attention Networks on reaction level property.
"""

import torch
import torch.nn as nn
from gnn.layer.hgatconv import HGATConv
from gnn.layer.readout import Set2SetThenCat
import dgl
import warnings
from gnn.utils import warn_stdout, np_split_by_size


class HGATReaction(nn.Module):
    """
    Heterograph attention network.


    Args:
        attn_mechanism (dict of dict): The attention mechanism, i.e. how the node
            features will be updated. The outer dict has `node types` as its key,
            and the inner dict has keys `nodes` and `edges`, where the values (list)
            of `nodes` are the `node types` that the master node will attend to,
            and the corresponding `edges` are the `edge types`.
        attn_order (list): The order to attend the node features. Order matters.
        in_feats (list): input feature size for the corresponding node in `attn_order`.
        num_gat_layers (int): number of graph attention layer
        gat_hidden_size (list): hidden size of graph attention layers
        num_heads (int): number of attention heads, the same for all nodes
        feat_drop (float, optional): [description]. Defaults to 0.0.
        attn_drop (float, optional): [description]. Defaults to 0.0.
        negative_slope (float, optional): [description]. Defaults to 0.2.
        gat_residual (bool, optional): [description]. Defaults to False.
        gat_activation (torch activation): activation fn of gat layers
        gat_batch_norm(bool): whether to apply batch norm to gat layer.
        readout_type (str): how to read out the features to bonds and then pass the fc
            layers. Options are {'bond', 'bond_cat_mean_max', 'bond_cat_mean_diff'}.
        num_fc_layers (int): number of fc layers. Note this is the number of hidden
            layers, i.e. there is an additional fc layer to map feature size to 1.
        fc_hidden_size (list): hidden size of fc layers
        fc_batch_norm (bool): whether to apply batch norm to fc layer
        fc_activation (torch activation): activation fn of fc layers
        fc_drop (float, optional): dropout ratio for fc layer.
        outdim (int): dimension of the output. For regression, choose 1 and for
            classification, set it to the number of classes.
    """

    def __init__(
        self,
        attn_mechanism,
        attn_order,
        in_feats,
        num_gat_layers=3,
        gat_hidden_size=[32, 64, 128],
        num_heads=4,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        gat_residual=True,
        gat_batch_norm=False,
        gat_activation="ELU",
        num_lstm_iters=5,
        num_lstm_layers=3,
        set2set_ntypes_direct=["global"],
        num_fc_layers=3,
        fc_hidden_size=[128, 64, 32],
        fc_batch_norm=False,
        fc_activation="ELU",
        fc_drop=0.0,
        outdim=1,
    ):
        super(HGATReaction, self).__init__()

        # activation fn
        if isinstance(gat_activation, str):
            gat_activation = getattr(nn, gat_activation)()
        if isinstance(fc_activation, str):
            fc_activation = getattr(nn, fc_activation)()

        self.gat_layers = nn.ModuleList()

        # input projection (no dropout)
        self.gat_layers.append(
            HGATConv(
                attn_mechanism=attn_mechanism,
                attn_order=attn_order,
                in_feats=in_feats,
                out_feats=gat_hidden_size[0],
                num_heads=num_heads,
                feat_drop=0.0,
                attn_drop=0.0,
                negative_slope=negative_slope,
                residual=gat_residual,
                batch_norm=gat_batch_norm,
                activation=gat_activation,
            )
        )

        # hidden gat layers
        for i in range(1, num_gat_layers):
            in_size = [gat_hidden_size[i - 1] * num_heads for _ in in_feats]
            self.gat_layers.append(
                HGATConv(
                    attn_mechanism=attn_mechanism,
                    attn_order=attn_order,
                    in_feats=in_size,
                    out_feats=gat_hidden_size[i],
                    num_heads=num_heads,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    residual=gat_residual,
                    batch_norm=gat_batch_norm,
                    activation=gat_activation,
                )
            )

        # set2set readout layer
        ntypes = ["atom", "bond"]
        in_size = [gat_hidden_size[-1] * num_heads for _ in attn_order]
        self.readout_layer = Set2SetThenCat(
            n_iters=num_lstm_iters,
            n_layer=num_lstm_layers,
            ntypes=ntypes,
            in_feats=in_size,
            ntypes_direct_cat=set2set_ntypes_direct,
        )

        # for atom and bond feat (# *2 because Set2Set used in Set2SetThenCat has out
        # feature twice the the size  of in feature)
        readout_out_size = (
            gat_hidden_size[-1] * num_heads * 2 + gat_hidden_size[-1] * num_heads * 2
        )
        # for global feat
        if set2set_ntypes_direct is not None:
            readout_out_size += gat_hidden_size[-1] * num_heads

        # need dropout?
        delta = 1e-3
        if fc_drop < delta:
            warnings.showwarning = warn_stdout
            warnings.warn(
                "`fc_drop = {}` provided for {} smaller than {}. "
                "Ignore dropout.".format(fc_drop, self.__class__.__name__, delta)
            )
            apply_drop = False
        else:
            apply_drop = True

        # fc layer to map to feature to bond energy
        self.fc_layers = nn.ModuleList()
        in_size = readout_out_size
        for i in range(num_fc_layers):
            out_size = fc_hidden_size[i]
            self.fc_layers.append(nn.Linear(in_size, out_size))
            # batch norm
            if fc_batch_norm:
                self.fc_layers.append(nn.BatchNorm1d(out_size))
            # activation
            self.fc_layers.append(fc_activation)
            # dropout
            if apply_drop:
                self.fc_layers.append(nn.Dropout(fc_drop))

            in_size = out_size

        # final output layer, mapping feature to the corresponding shape
        self.fc_layers.append(nn.Linear(in_size, outdim))

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
        for layer in self.gat_layers:
            feats = layer(graph, feats)

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
