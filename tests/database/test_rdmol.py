from gnn.database.rdmol import fragment_rdkit_mol
from .utils import create_LiEC_rdkit_mol


def test_fragment_rdkit_mol():

    rdm = create_LiEC_rdkit_mol()
    # print("#" * 80, "original", Chem.MolToMolBlock(rdm, forceV3000=True))

    # break C C bond in the ring
    bond = (2, 5)
    fragments = fragment_rdkit_mol(rdm, bonds=[bond])
    mols = fragments[bond]
    assert len(mols) == 1
    assert len(mols[0].GetBonds()) == 10

    # print("#" * 80, "C-C", Chem.MolToMolBlock(mols[0], forceV3000=True))

    # break the Li O bonds
    bond = (3, 6)
    fragments = fragment_rdkit_mol(rdm, bonds=[bond])
    mols = fragments[bond]
    assert len(mols) == 2
    assert len(mols[0].GetBonds()) == 10 and len(mols[1].GetBonds()) == 0

    # print("#" * 80, "Li O", Chem.MolToMolBlock(mols[0], forceV3000=True))
    # print("#" * 80, "Li O", Chem.MolToMolBlock(mols[1], forceV3000=True))

    # # break C=O bond, generating two fragments
    bond = (3, 1)
    fragments = fragment_rdkit_mol(rdm, bonds=[bond])
    mols = fragments[bond]
    assert len(mols) == 2
    assert len(mols[0].GetBonds()) == 9 and len(mols[1].GetBonds()) == 1

    # print("#" * 80, "C O", Chem.MolToMolBlock(mols[0], forceV3000=True))
    # print("#" * 80, "C O", Chem.MolToMolBlock(mols[1], forceV3000=True))
