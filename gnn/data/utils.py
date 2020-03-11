import warnings
import itertools
from rdkit import Chem
from collections import defaultdict, OrderedDict
from gnn.utils import expand_path, yaml_dump


def graph_struct_list_representation(g):
    """
    Represent the heterogrpah canonical edges as a dict.

    Args:
        g: a dgl heterograph

    Returns:
        dict of dict of list: dest node is the outer key; `nodes`and `edges` are the
        innter keys; src nodes are the values associated with `nodes` and edge types
        are the vlues associated with `edges`.

    Example:
        Suppose the graph has canonical edges:
         src   edge   dest

        [['A', 'A2B', 'B'],
         ['C', 'C2B', 'B'],
         ['C', 'C2A', 'A']]

        This function rturns:
        {
         'A':{'nodes':['C'],
              'edges':['C2A']},
         'B':{'nodes':['A', 'C'],
              'edges':['A2B', 'C2B']}
        }
    """
    graph_strcut = {
        t: {"nodes": defaultdict(list), "edges": defaultdict(list)} for t in g.ntypes
    }
    for src, edge, dest in g.canonical_etypes:
        graph_strcut[dest]["nodes"].append(src)
        graph_strcut[dest]["edges"].append(edge)

    return graph_strcut


def get_bond_to_atom_map(g):
    """
    Query which atoms are associated with the bonds.

    Args:
        g: dgl heterograph

    Returns:
        dict: with bond index as the key and a tuple of atom indices of atoms that
            form the bond.
    """
    nbonds = g.number_of_nodes("bond")
    bond_to_atom_map = dict()
    for i in range(nbonds):
        atoms = g.successors(i, "b2a")
        bond_to_atom_map[i] = sorted(atoms)
    return bond_to_atom_map


def get_atom_to_bond_map(g):
    """
    Query which bonds are associated with the atoms.

    Args:
        g: dgl heterograph

    Returns:
        dict: with atom index as the key and a list of indices of bonds is
        connected to the atom.
    """
    natoms = g.number_of_nodes("atom")
    atom_to_bond_map = dict()
    for i in range(natoms):
        bonds = g.successors(i, "a2b")
        atom_to_bond_map[i] = sorted(list(bonds))
    return atom_to_bond_map


def get_dataset_species(filename):
    """
    Get all the species of atoms appearing in the dataset.

    Args:
        filename (str): sdf file name

    Returns:
        list: a sequence of species string
    """
    suppl = Chem.SDMolSupplier(filename, sanitize=True, removeHs=False)
    system_species = set()
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        atoms = mol.GetAtoms()
        species = [a.GetSymbol() for a in atoms]
        system_species.update(species)

    return list(system_species)


def create_edge_label_based_on_bond(
    filename, sdf_filename="mols.sdf", label_filename="bond_label.yaml"
):
    """
    For a molecule from SDF file, creating complete graph for atoms and label the edges
    based on whether its an actual bond or not.

    The order of the edges are (0,1), (0,2), ... , (0, N-1), (1,2), (1,3), ...,
    (N-2, N-1), where N is the number of atoms.

    Args:
        filename (str): name of the input sdf file
        sdf_filename (str): name of the output sdf file
        label_filename (str): name of the output label file
    """

    def read_sdf(filename):
        filename = expand_path(filename)
        supp = Chem.SDMolSupplier(filename, sanitize=True, removeHs=False)
        all_mols = []
        for i, mol in enumerate(supp):
            if mol is None:
                print("bad mol:", i)
            else:
                all_mols.append(mol)
        print("{} molecules read from sdf file".format(len(all_mols)))
        return all_mols

    def get_bond_label(m):
        """
        Get to know whether an edge in a complete graph is a bond.

        Returns:
            OrderedDict: with bond index as key and a bool indicating whether it's a
                bond as value.

        """

        num_bonds = m.GetNumBonds()
        if num_bonds < 1:
            warnings.warn("molecular has no bonds")

        num_atoms = m.GetNumAtoms()
        bond_label = OrderedDict()
        for u, v in itertools.combinations(range(num_atoms), 2):
            bond = m.GetBondBetweenAtoms(u, v)
            if bond is not None:
                bond_label[tuple(sorted([u, v]))] = True
            else:
                bond_label[tuple(sorted([u, v]))] = False

        return bond_label

    mols = read_sdf(filename)

    labels = []
    with open(sdf_filename, "w") as f:
        for m in mols:
            labels.append(get_bond_label(m))
            sdf = Chem.MolToMolBlock(m)
            f.write(sdf + "$$$$\n")
    yaml_dump(labels, label_filename)
