from gnn.model.hgat_reaction_network import mol_graph_to_rxn_graph
from gnn.model.gated_mol import GatedGCNMol


class GatedGCNReactionNetwork(GatedGCNMol):
    def forward(self, graph, feats, reactions, norm_atom, norm_bond):
        """
        Args:
            graph (DGLHeteroGraph or BatchedDGLHeteroGraph): (batched) molecule graphs
            feats (dict): node features with node type as key and the corresponding
                features as value.
            reactions (list): a sequence of :class:`gnn.data.reaction_network.Reaction`,
                each representing a reaction.
            norm_atom (2D tensor)
            norm_bond (2D tensor)

        Returns:
            2D tensor: of shape(N, M), where `M = outdim`.
        """

        # embedding
        feats = self.embedding(feats)

        # gated layer
        for layer in self.gated_layers:
            feats = layer(graph, feats, norm_atom, norm_bond)

        # convert mol graphs to reaction graphs by subtracting reactant feats from
        # products feats
        graph, feats = mol_graph_to_rxn_graph(graph, feats, reactions)

        # readout layer
        feats = self.readout_layer(graph, feats)

        # fc
        for layer in self.fc_layers:
            feats = layer(feats)

        return feats

    def feature_before_fc(self, graph, feats, reactions, norm_atom, norm_bond):
        """
        This is used when we want to visualize feature.
        """
        # embedding
        feats = self.embedding(feats)

        # gated layer
        for layer in self.gated_layers:
            feats = layer(graph, feats, norm_atom, norm_bond)

        # convert mol graphs to reaction graphs by subtracting reactant feats from
        # products feats
        graph, feats = mol_graph_to_rxn_graph(graph, feats, reactions)

        # readout layer
        feats = self.readout_layer(graph, feats)

        return feats
