from gnn.database.reaction import ReactionExtractor, ReactionsOfSameBond
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


def test_reactions_with_same_bonds():
    def assert_mol(m, formula, charge):
        assert m.formula == formula
        assert m.charge == charge

    _, A2BC = create_reactions()
    A2BC = A2BC[:2]  # break same bonds

    reactant = A2BC[0].reactants[0]
    rsb = ReactionsOfSameBond(reactant)
    for rxn in A2BC:
        rsb.add(rxn)

    # generating complete reactions
    comp_rxns = rsb.create_complement_reactions()
    assert len(comp_rxns) == 1
    rxn = comp_rxns[0]
    assert_mol(rxn.reactants[0], "C3H1", 0)
    assert_mol(rxn.products[0], "H1", -1)
    assert_mol(rxn.products[1], "C3", 1)

    # order reactions
    ordered_rxns = rsb.order_reactions(complement_reactions=True)
    assert len(ordered_rxns) == 3
    assert ordered_rxns[0] == A2BC[0]
    assert ordered_rxns[1] == A2BC[1]
    assert ordered_rxns[2] == comp_rxns[0]
