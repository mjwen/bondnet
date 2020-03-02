from gnn.database.reaction import (
    ReactionExtractor,
    ReactionsOfSameBond,
    ReactionsMultiplePerBond,
)
from .utils import create_reactions, create_molecules


def test_reaction_atom_mapping():
    # {0:3} first because products are ordered
    ref_mapping = [{0: 3}, {0: 0, 1: 2, 2: 1}]

    _, A2BC = create_reactions()

    # m0 to m2 m4
    reaction = A2BC[0]
    mapping = reaction.atom_mapping()

    assert mapping == ref_mapping


def test_reaction_bond_mapping_single_index():

    ref_m0_to_m1_mapping = [{0: 1, 1: 2, 2: 3}]
    # {} first because products are ordered
    ref_m0_to_m2_m4_mapping = [{}, {0: 1, 1: 0, 2: 2}]
    ref_m7_to_m4_m4_mapping = [{}, {}]

    A2B, A2BC = create_reactions()

    # m1 to m2
    assert A2B[0].bond_mapping_by_single_index() == ref_m0_to_m1_mapping

    # m0 to m2 m4
    assert A2BC[0].bond_mapping_by_single_index() == ref_m0_to_m2_m4_mapping

    # m7 to m4 m4
    assert A2BC[3].bond_mapping_by_single_index() == ref_m7_to_m4_m4_mapping


def test_reaction_bond_mapping_tuple_index():
    ref_m0_to_m1_mapping = [{(0, 2): (0, 2), (1, 2): (1, 2), (2, 3): (2, 3)}]
    # {} first because products are ordered
    ref_m0_to_m2_m4_mapping = [{}, {(0, 1): (0, 2), (0, 2): (0, 1), (1, 2): (1, 2)}]
    ref_m7_to_m4_m4_mapping = [{}, {}]

    A2B, A2BC = create_reactions()

    # m1 to m2
    assert A2B[0].bond_mapping_by_tuple_index() == ref_m0_to_m1_mapping

    # m0 to m2 m4
    assert A2BC[0].bond_mapping_by_tuple_index() == ref_m0_to_m2_m4_mapping

    # m7 to m4 m4
    assert A2BC[3].bond_mapping_by_tuple_index() == ref_m7_to_m4_m4_mapping


def test_extract_reactions():
    ref_A2B, ref_A2BC = create_reactions()

    mols = create_molecules()
    extractor = ReactionExtractor(mols)
    A2B = extractor.extract_A_to_B_style_reaction()
    A2BC = extractor.extract_A_to_B_C_style_reaction()

    # reactions is not hashable, since essentially its products (or reactants) is a set
    # and set is not hashable. As a results, we cannot make a set of reactions and
    # compare them
    assert (len(A2B)) == 1
    for rxn in ref_A2B:
        assert rxn in A2B
    for rxn in A2B:
        assert rxn in ref_A2B

    assert (len(A2BC)) == 4
    for rxn in ref_A2BC:
        assert rxn in A2BC
    for rxn in A2BC:
        assert rxn in ref_A2BC


def test_reactions_of_same_bond():
    def assert_mol(m, formula, charge):
        assert m.formula == formula
        assert m.charge == charge

    _, A2BC = create_reactions()
    A2BC = A2BC[:2]  # break same bonds

    reactant = A2BC[0].reactants[0]
    rsb = ReactionsOfSameBond(reactant, A2BC)

    # test create complement reactions
    comp_rxns = rsb.create_complement_reactions()
    assert len(comp_rxns) == 1
    rxn = comp_rxns[0]
    assert_mol(rxn.reactants[0], "C2H1O1", 0)
    assert_mol(rxn.products[0], "H1", -1)
    assert_mol(rxn.products[1], "C2O1", 1)

    # test order reactions
    ordered_rxns = rsb.order_reactions(complement_reactions=False)
    assert len(ordered_rxns) == 2
    assert ordered_rxns[0] == A2BC[0]
    assert ordered_rxns[1] == A2BC[1]

    ordered_rxns = rsb.order_reactions(complement_reactions=True)
    assert len(ordered_rxns) == 3
    assert ordered_rxns[0] == A2BC[0]
    assert ordered_rxns[1] == A2BC[1]
    assert ordered_rxns[2] == comp_rxns[0]


def test_reactions_multiple_per_bond():

    A2B, A2BC = create_reactions()
    # of same reactant
    reactions = A2B + A2BC[:2]

    reactant = reactions[0].reactants[0]
    rmb = ReactionsMultiplePerBond(reactant, reactions)

    # test group by bond
    rsb_group = rmb.group_by_bond()
    for rsb in rsb_group:
        if rsb.broken_bond == (0, 1):
            assert len(rsb.reactions) == 1
            assert rsb.reactions[0] == A2B[0]
        elif rsb.broken_bond == (2, 3):
            assert len(rsb.reactions) == 2
            assert rsb.reactions[0] == A2BC[0]
            assert rsb.reactions[1] == A2BC[1]
        else:
            assert rsb.reactions == []

    # test order reactions
    ordered_rxns = rmb.order_reactions(complement_reactions=False)
    assert len(ordered_rxns) == 3
    assert ordered_rxns[0] == A2B[0]
    assert ordered_rxns[1] == A2BC[0]
    assert ordered_rxns[2] == A2BC[1]

    ordered_rxns = rmb.order_reactions(complement_reactions=True)
    assert len(ordered_rxns) == 6  # 1 for each C-C bond in the ring, 3 for C-H bond
    assert ordered_rxns[0] == A2B[0]
    assert ordered_rxns[1] == A2BC[0]
    assert ordered_rxns[2] == A2BC[1]
