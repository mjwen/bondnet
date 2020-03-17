import numpy as np
from gnn.data.reaction_network import Reaction, ReactionNetwork


class TestReaction:
    def test_mapping_as_list(self):
        ref = [3, 0, 2, 1]
        mappings = [{0: 1, 1: 3}, {0: 2, 1: 0}]
        mp_list = Reaction._mapping_as_list(mappings)

        assert mp_list == ref


class TestReactionNetwork:
    def test_get_molecules_in_reactions(self):

        molecules = ["m0", "m1", "m2", "m3", "m4"]
        rxn1 = Reaction(reactants=[2], products=[0, 4])
        rxn2 = Reaction(reactants=[1], products=[4])

        rn = ReactionNetwork(molecules, [rxn1, rxn2])

        sub_rxns, sub_mols = rn.subselect_reactions(indices=[0, 1])
        assert np.array_equal(sub_mols, ["m0", "m1", "m2", "m4"])

        ref_reactants = [[2], [1]]
        ref_products = [[0, 3], [3]]
        for i, rxn in enumerate(sub_rxns):
            assert rxn.reactants == ref_reactants[i]
            assert rxn.products == ref_products[i]
