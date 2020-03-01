from gnn.database.database import MoleculeWrapperFromAtomsAndBonds
from gnn.database.reaction import Reaction


def create_molecules():
    """
    Create a list of molecules:

    m0: charge 0
        C 0
       / \
      /___\
     O     C---H
     1     2   3

    m1: charge 0
        C 0
         \
       ___\
     O     C---H
     1     2   3

    m2: charge 0 (note the atom index order between this and m0)
        C 0
       / \
      /___\
     O     C
     2     1

    m2: charge -1 (note the atom index order between this and m0)
        C 0
       / \
      /___\
     O     C
     2     1

    m4: charge 0
    H

    m5: charge 1
    H

    m6: charge -1
    H

    m7: charge 0
    H--H


    The below reactions exists (w.r.t. graph connectivity and charge):

    m0 -> m1    C2HO (0) -> C2HO (0)
    m0 -> m2 + m4   C2HO (0) -> C2O (0) + H (0)
    m0 -> m3 + m5   C2HO (0) -> C2O (-1) + H (1)
    m7 -> m4 + m4   H2 (0) -> H (0) + H (0)
    m7 -> m5 + m6   H2 (0) -> H (1) + H (-1)
    """

    mols = []

    # m0, charge 0
    species = ["C", "O", "C", "H"]
    coords = [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
    ]
    charge = 0
    bonds = [(0, 1), (0, 2), (1, 2), (2, 3)]
    mols.append(
        MoleculeWrapperFromAtomsAndBonds(
            species=species,
            coords=coords,
            charge=charge,
            bonds=bonds,
            mol_id="m0",
            free_energy=0.0,
        )
    )

    # m1, same as m0 but no bond between atoms 0 and 1, charge 0
    bonds = [(0, 2), (1, 2), (2, 3)]
    mols.append(
        MoleculeWrapperFromAtomsAndBonds(
            species=species,
            coords=coords,
            charge=charge,
            bonds=bonds,
            mol_id="m1",
            free_energy=1.0,
        )
    )

    # m2, m0 without H, charge 0
    species = ["C", "C", "O"]
    coords = [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    charge = 0
    bonds = [(0, 1), (0, 2), (1, 2)]
    mols.append(
        MoleculeWrapperFromAtomsAndBonds(
            species=species,
            coords=coords,
            charge=charge,
            bonds=bonds,
            mol_id="m2",
            free_energy=2.0,
        )
    )

    # m3, m0 without H, charge -1
    charge = -1
    mols.append(
        MoleculeWrapperFromAtomsAndBonds(
            species=species,
            coords=coords,
            charge=charge,
            bonds=bonds,
            mol_id="m3",
            free_energy=3.0,
        )
    )

    # m4, H, charge 0
    species = ["H"]
    coords = [[1.0, 0.0, 1.0]]
    charge = 0
    bonds = []
    mols.append(
        MoleculeWrapperFromAtomsAndBonds(
            species=species,
            coords=coords,
            charge=charge,
            bonds=bonds,
            mol_id="m4",
            free_energy=4.0,
        )
    )

    # m5, H, charge 1
    charge = 1
    mols.append(
        MoleculeWrapperFromAtomsAndBonds(
            species=species,
            coords=coords,
            charge=charge,
            bonds=bonds,
            mol_id="m5",
            free_energy=5.0,
        )
    )

    # m6, H, charge -1
    charge = -1
    mols.append(
        MoleculeWrapperFromAtomsAndBonds(
            species=species,
            coords=coords,
            charge=charge,
            bonds=bonds,
            mol_id="m6",
            free_energy=6.0,
        )
    )

    # m7, H2, charge 0
    species = ["H", "H"]
    coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    charge = 0
    bonds = [(0, 1)]
    mols.append(
        MoleculeWrapperFromAtomsAndBonds(
            species=species,
            coords=coords,
            charge=charge,
            bonds=bonds,
            mol_id="m7",
            free_energy=7.0,
        )
    )

    return mols


def create_reactions():
    """
    Create a list of reactions, using the mols returned by create_mols.
    """
    mols = create_molecules()
    A2B = [Reaction(reactants=[mols[0]], products=[mols[1]], broken_bond=(0, 1))]
    A2BC = [
        Reaction(reactants=[mols[0]], products=[mols[2], mols[4]], broken_bond=(2, 3)),
        Reaction(reactants=[mols[0]], products=[mols[3], mols[5]], broken_bond=(2, 3)),
        Reaction(reactants=[mols[7]], products=[mols[4], mols[4]], broken_bond=(0, 1)),
        Reaction(reactants=[mols[7]], products=[mols[5], mols[6]], broken_bond=(0, 1)),
    ]

    return A2B, A2BC
