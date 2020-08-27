def get_dataset_species(molecules):
    """
    Get all the species of atoms appearing in the the molecules.

    Args:
        molecules (list): rdkit molecules

    Returns:
        list: a sequence of species string
    """
    system_species = set()
    for mol in molecules:
        if mol is None:
            continue
        species = [a.GetSymbol() for a in mol.GetAtoms()]
        system_species.update(species)

    return sorted(system_species)
