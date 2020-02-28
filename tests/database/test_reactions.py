from gnn.database.reaction import (
    ReactionExtractor,
    ReactionsOfSameBond,
    ReactionsMultiplePerBond,
)
from .utils import create_reactions, create_molecules


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
    assert_mol(rxn.reactants[0], "C3H1", 0)
    assert_mol(rxn.products[0], "H1", -1)
    assert_mol(rxn.products[1], "C3", 1)

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
