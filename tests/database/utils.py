from gnn.database.database import MoleculeWrapperFromAtomsAndBonds


def create_list_of_molecules():
    """
    Create a list of molecules, such that the below reactions exists (w.r.t. graph
    connectivity and charge):

    m0 -> m1    C3H (0) -> C3H (0)
    m0 -> m2 + m4   C3H (0) -> C3 (0) + H (0)
    m0 -> m3 + m5   C3H (0) -> C3 (-1) + H (1)
    m7 -> m4 + m4   H2 (0) -> H (0) + H (0)
    m7 -> m5 + m6   H2 (0) -> H (1) + H (-1)
    """

    #
    #           C 0
    #          / \
    #         /___\
    #        C     C---H
    #        1     2   3

    mols = []

    # m0, charge 0
    species = ["C", "C", "C", "H"]
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
            species=species, coords=coords, charge=charge, bonds=bonds, mol_id="m0"
        )
    )

    # m1, same as m0 but no bond between atoms 0 and 1, charge 0
    bonds = [(0, 2), (1, 2), (2, 3)]
    mols.append(
        MoleculeWrapperFromAtomsAndBonds(
            species=species, coords=coords, charge=charge, bonds=bonds, mol_id="m1"
        )
    )

    # m2, m0 without H, charge 0
    species = ["C", "C", "C"]
    coords = [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    charge = 0
    bonds = [(0, 1), (0, 2), (1, 2)]
    mols.append(
        MoleculeWrapperFromAtomsAndBonds(
            species=species, coords=coords, charge=charge, bonds=bonds, mol_id="m4"
        )
    )

    # m3, m0 without H, charge -1
    charge = -1
    mols.append(
        MoleculeWrapperFromAtomsAndBonds(
            species=species, coords=coords, charge=charge, bonds=bonds, mol_id="m5"
        )
    )

    # m4, H, charge 0
    species = ["H"]
    coords = [[1.0, 0.0, 1.0]]
    charge = 0
    bonds = []
    mols.append(
        MoleculeWrapperFromAtomsAndBonds(
            species=species, coords=coords, charge=charge, bonds=bonds, mol_id="m2"
        )
    )

    # m5, H, charge 1
    charge = 1
    mols.append(
        MoleculeWrapperFromAtomsAndBonds(
            species=species, coords=coords, charge=charge, bonds=bonds, mol_id="m3"
        )
    )

    # m6, H, charge -1
    charge = -1
    mols.append(
        MoleculeWrapperFromAtomsAndBonds(
            species=species, coords=coords, charge=charge, bonds=bonds, mol_id="m3"
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
            species=species, coords=coords, charge=charge, bonds=bonds, mol_id="m6"
        )
    )

    return mols
