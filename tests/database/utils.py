from gnn.database.molwrapper import MoleculeWrapperFromAtomsAndBonds
from gnn.database.reaction import Reaction


def create_C2H4O1():
    r"""
                O(0)
               / \
              /   \
      H(1)--C(2)--C(3)--H(4)
             |     |
            H(5)  H(6)
    """
    species = ["O", "H", "C", "C", "H", "H", "H"]
    coords = [
        [0.0, 1.0, 0.0],
        [-2.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ]
    charge = 0
    bonds = [(0, 2), (0, 3), (1, 2), (2, 3), (3, 4), (2, 5), (3, 6)]
    m = MoleculeWrapperFromAtomsAndBonds(
        species=species,
        coords=coords,
        charge=charge,
        bonds=bonds,
        mol_id="mol",
        free_energy=0.0,
    )

    return m


def create_symmetric_molecules():
    r"""
    Create a list of molecules, which can form reactions where the reactant is symmetric.

    m0: charge 0
    H(0)---C(1)---H(2)
           / \
          /   \
       O(3)---O(4)

    m1: charge 0
    H(0)---C(1)---H(2)
           / \
          /   \
       O(3)   O(4)

    m2: charge 0
    H(0)---C(1)
           /  \
          /   \
       O(2)---O(3)

    m3: charge -1
    H(0)---C(1)
           /  \
          /   \
       O(2)---O(3)

    m4: charge 0
    H

    m5: charge 1
    H

    m6: charge -1
    H

    m7: charge 0
    H--H

    The below reactions exists (w.r.t. graph connectivity and charge):

    A -> B:
    m0 -> m1        C1H2O2 (0) -> C1H2O2 (0) + H (0)
    A -> B+C:
    m0 -> m2 + m4   C1H2O2 (0) -> C1H1O2 (0) + H (0)    # breaking bond (1,2)
    m0 -> m3 + m5   C1H2O2 (0) -> C1H1O2 (0) + H (0)    # breaking bond (1,2)
    m0 -> m2 + m4   C1H2O2 (0) -> C1H1O2 (0) + H (0)    # breaking bond (0,1)
    m0 -> m3 + m5   C1H2O2 (0) -> C1H1O2 (0) + H (0)    # breaking bond (0,1)
    m7 -> m4 + m4   H2 (0) -> H (0) + H (0)
    m7 -> m5 + m6   H2 (0) -> H (1) + H (-1)
    """

    mols = []

    # m0, charge 0
    species = ["H", "C", "H", "O", "O"]
    coords = [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
    charge = 0
    bonds = [(0, 1), (1, 2), (1, 3), (1, 4), (3, 4)]
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

    # m1, charge 0
    bonds = [(0, 1), (1, 2), (1, 3), (1, 4)]
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

    # m2, charge 0
    species = ["H", "C", "O", "O"]
    coords = [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
    charge = 0
    bonds = [(0, 1), (1, 2), (1, 3), (2, 3)]
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

    # m3, charge -1
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
    coords = [[1.0, 0.0, 0.0]]
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


def create_nonsymmetric_molecules():
    r"""
    Create a list of molecules, which can form reactions where the reactant is
    nonsymmetric.

    m0: charge 0
        C(0)
     0 /  \  1
      /____\     3
    O(1) 2 N(2)---H(3)

    m1: charge 0 (note the atom index order between this and m0)
        C(0)
          \ 0
       ____\      2
    O(2) 1  N(1)---H(3)

    m2: charge 0 (note the atom index order between this and m0)
        C(0)
    1  /  \ 0
      /____\
    O(2) 2  N(1)

    m3: charge 0
    H(0)


    The below reactions exists (w.r.t. graph connectivity and charge):

    m0 -> m1    CHNO (0) -> CHNO (0)
    m0 -> m2 + m3   CHNO (0) -> CNO (0) + H (0)
    """

    mols = []

    # m0
    species = ["C", "O", "N", "H"]
    coords = [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
    ]
    bonds = [(0, 1), (0, 2), (1, 2), (2, 3)]
    mols.append(
        MoleculeWrapperFromAtomsAndBonds(
            species=species,
            coords=coords,
            charge=0,
            bonds=bonds,
            mol_id="m0",
            free_energy=0.0,
        )
    )

    # m1
    species = ["C", "N", "O", "H"]
    coords = [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
    ]
    bonds = [(0, 1), (1, 2), (1, 3)]
    mols.append(
        MoleculeWrapperFromAtomsAndBonds(
            species=species,
            coords=coords,
            charge=0,
            bonds=bonds,
            mol_id="m1",
            free_energy=1.0,
        )
    )

    # m2, m0 without H
    species = ["C", "N", "O"]
    coords = [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    bonds = [(0, 1), (0, 2), (1, 2)]
    mols.append(
        MoleculeWrapperFromAtomsAndBonds(
            species=species,
            coords=coords,
            charge=0,
            bonds=bonds,
            mol_id="m2",
            free_energy=2.0,
        )
    )

    # m3, H
    species = ["H"]
    coords = [[1.0, 0.0, 1.0]]
    bonds = []
    mols.append(
        MoleculeWrapperFromAtomsAndBonds(
            species=species,
            coords=coords,
            charge=0,
            bonds=bonds,
            mol_id="m3",
            free_energy=3.0,
        )
    )

    return mols


def create_reactions_symmetric_reactant():
    """
    Create a list of reactions, using the mols returned by `create_symmetric_molecules`.
    """
    mols = create_symmetric_molecules()
    A2B = [Reaction(reactants=[mols[0]], products=[mols[1]], broken_bond=(3, 4))]
    A2BC = [
        Reaction(reactants=[mols[0]], products=[mols[2], mols[4]], broken_bond=(1, 2)),
        Reaction(reactants=[mols[0]], products=[mols[3], mols[5]], broken_bond=(1, 2)),
        Reaction(reactants=[mols[0]], products=[mols[2], mols[4]], broken_bond=(0, 1)),
        Reaction(reactants=[mols[0]], products=[mols[3], mols[5]], broken_bond=(0, 1)),
        Reaction(reactants=[mols[7]], products=[mols[4], mols[4]], broken_bond=(0, 1)),
        Reaction(reactants=[mols[7]], products=[mols[5], mols[6]], broken_bond=(0, 1)),
    ]

    return A2B, A2BC


def create_reactions_nonsymmetric_reactant():
    """
    Create a list of reactions, using the mols returned by
    `create_nonsymmetric_molecules`.
    """
    mols = create_nonsymmetric_molecules()
    A2B = [Reaction(reactants=[mols[0]], products=[mols[1]], broken_bond=(0, 1))]
    A2BC = [
        Reaction(reactants=[mols[0]], products=[mols[2], mols[3]], broken_bond=(2, 3))
    ]

    return A2B, A2BC
