from gnn.database.reaction import Reaction
from gnn.database.reaction import ReactionExtractor

from .utils import create_list_of_molecules


def test_extract_reactions():

    mols = create_list_of_molecules()
    ref_A2B = [Reaction(reactants=[mols[0]], products=[mols[1]])]
    ref_A2BC = [
        Reaction(reactants=[mols[0]], products=[mols[2], mols[4]]),
        Reaction(reactants=[mols[0]], products=[mols[3], mols[5]]),
        Reaction(reactants=[mols[7]], products=[mols[4], mols[4]]),
        Reaction(reactants=[mols[7]], products=[mols[5], mols[6]]),
    ]

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
