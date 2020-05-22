import copy
import logging
import warnings
import numpy as np
import itertools
import networkx as nx
import pymatgen
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from gnn.database.rdmol import create_rdkit_mol_from_mol_graph
from gnn.utils import create_directory, expand_path, yaml_dump

logger = logging.getLogger(__name__)


class MoleculeWrapper:
    """
    A wrapper of pymatgen Molecule, MoleculeGraph, rdkit Chem.Mol... to make it
    easier to use molecules.

    Arguments:
        mol_graph (MoleculeGraph): pymatgen molecule graph instance
        free_energy (float): free energy of the molecule
        id (str): (unique) identification of the molecule
    """

    def __init__(self, mol_graph, free_energy=None, id=None):
        self.mol_graph = mol_graph
        self.pymatgen_mol = mol_graph.molecule
        self.free_energy = free_energy
        self.id = id

        # set when corresponding method is called
        self._rdkit_mol = None
        self._fragments = None
        self._isomorphic_bonds = None

    @property
    def charge(self):
        """
        Returns:
            int: charge of the molecule
        """
        return self.pymatgen_mol.charge

    @property
    def formula(self):
        """
        Returns:
            str: chemical formula of the molecule, e.g. H2CO3.
        """
        return self.pymatgen_mol.composition.alphabetical_formula.replace(" ", "")

    @property
    def composition_dict(self):
        """
        Returns:
            dict: with chemical species as key and number of the species as value.
        """
        d = self.pymatgen_mol.composition.as_dict()
        return {k: int(v) for k, v in d.items()}

    @property
    def weight(self):
        """
        Returns:
            int: molecule weight
        """
        return self.pymatgen_mol.composition.weight

    @property
    def num_atoms(self):
        """
        Returns:
            int: number of atoms in molecule
        """
        return len(self.pymatgen_mol)

    @property
    def species(self):
        """
        Species of atoms. Order is the same as self.atoms.
        Returns:
            list: Species string.
        """
        return [str(s) for s in self.pymatgen_mol.species]

    @property
    def coords(self):
        """
        Returns:
            2D array: of shape (N, 3) where N is the number of atoms.
        """
        return np.asarray(self.pymatgen_mol.cart_coords)

    @property
    def bonds(self):
        """
        Returns:
            dict: with bond index (a tuple of atom indices) as the key and and bond
                attributes as the value.
        """
        return {tuple(sorted([i, j])): attr for i, j, attr in self.graph.edges.data()}

    @property
    def graph(self):
        """
        Returns:
            networkx graph used by mol_graph
        """
        return self.mol_graph.graph

    @property
    def rdkit_mol(self):
        """
        Returns:
            rdkit molecule
        """
        if self._rdkit_mol is None:
            self._rdkit_mol, _ = create_rdkit_mol_from_mol_graph(
                self.mol_graph, name=str(self), force_sanitize=False
            )
        return self._rdkit_mol

    @rdkit_mol.setter
    def rdkit_mol(self, m):
        self._rdkit_mol = m

    @property
    def fragments(self):
        """
        Get fragments of the molecule by breaking all the bonds.

        Returns:
            A dictionary with bond index (a tuple (idx1, idx2)) as key, and a list
            of the mol_graphs of the fragments as value (each list is of size 1 or 2).
            The dictionary is empty if the mol has no bonds.
        """
        if self._fragments is None:
            bonds = [b for b, _ in self.bonds.items()]
            self._fragments = fragment_mol_graph(self.mol_graph, bonds)
        return self._fragments

    @property
    def isomorphic_bonds(self):
        r"""
        Find isomorphic bonds. Isomorphic bonds are defined as bonds that the same
        fragments (in terms of fragment connectivity) are obtained by breaking the bonds
        separately.

        For example, for molecule

               0     1
            H1---C0---H2
              2 /  \ 3
              O3---O4

        bond 0 is isomorphic to bond 1 and bond 2 is isomorphic to bond 3.

        Returns:
            list of list: each inner list contains the indices (tuple) of bonds that
                are isomorphic. For the above example, this function
                returns [[(0,1), (0,2)], [(0,3), (0,4)]]
        """

        if self._isomorphic_bonds is None:
            iso_bonds = []
            for b1, b2 in itertools.combinations(self.fragments, 2):
                frags1 = self.fragments[b1]
                frags2 = self.fragments[b2]

                if len(frags1) == len(frags2) == 1:
                    if frags1[0].isomorphic_to(frags2[0]):
                        iso_bonds.append([b1, b2])
                elif len(frags1) == len(frags2) == 2:
                    if (
                        frags1[0].isomorphic_to(frags2[0])
                        and frags1[1].isomorphic_to(frags2[1])
                    ) or (
                        frags1[0].isomorphic_to(frags2[1])
                        and frags1[1].isomorphic_to(frags2[0])
                    ):
                        iso_bonds.append([b1, b2])

            res = []
            for b1, b2 in iso_bonds:
                find_b1_or_b2 = False
                for group in res:
                    if b1 in group or b2 in group:
                        group.extend([b1, b2])
                        find_b1_or_b2 = True
                        break
                if not find_b1_or_b2:
                    group = [b1, b2]
                    res.append(group)

            # remove duplicate in each group
            res = [list(set(g)) for g in res]
            self._isomorphic_bonds = res

        return self._isomorphic_bonds

    def get_sdf_bond_indices(self, zero_based=False, sdf=None):
        """
        Get the indices of bonds as specified in the sdf file.

        zero_based (bool): If True, the atom index will be converted to zero based.
        sdf (str): the sdf string for parsing. If None, it is created from the mol.

        Returns:
            list of tuple: each tuple specifies a bond.
        """
        sdf = sdf or self.write()

        lines = sdf.split("\n")
        start = end = 0
        for i, ln in enumerate(lines):
            if "BEGIN BOND" in ln:
                start = i + 1
            if "END BOND" in ln:
                end = i
                break

        bonds = [
            tuple(sorted([int(i) for i in ln.split()[4:6]])) for ln in lines[start:end]
        ]

        if zero_based:
            bonds = [(b[0] - 1, b[1] - 1) for b in bonds]

        return bonds

    def get_sdf_bond_indices_v2000(self, sdf=None):
        """
        Get the indices of bonds as specified in the sdf file.

        Returns:
            list of tuple: each tuple specifies a bond.
        """
        sdf = sdf or self.write(v3000=False)
        lines = sdf.split("\n")
        split_3 = lines[3].split()
        natoms = int(split_3[0])
        nbonds = int(split_3[1])
        bonds = []
        for line in lines[4 + natoms : 4 + natoms + nbonds]:
            bonds.append(tuple(sorted([int(i) for i in line.split()[:2]])))
        return bonds

    def subgraph_atom_mapping(self, bond):
        """
        Find the atoms in the two subgraphs by breaking a bond in a molecule.

        Returns:
            tuple of list: each list contains the atoms in one subgraph.
        """

        original = copy.deepcopy(self.mol_graph)
        original.break_edge(bond[0], bond[1], allow_reverse=True)

        # A -> B breaking
        if nx.is_weakly_connected(original.graph):
            mapping = list(range(self.num_atoms))
            return mapping, mapping
        # A -> B + C breaking
        else:
            components = nx.weakly_connected_components(original.graph)
            nodes = [original.graph.subgraph(c).nodes for c in components]
            mapping = tuple([sorted(list(n)) for n in nodes])
            if len(mapping) != 2:
                raise Exception("Mol not split into two parts")
            return mapping

    def write(self, filename=None, name=None, format="sdf", v3000=True):
        """Write a molecule to file or as string using rdkit.

        Args:
            filename (str): name of the file to write the output. If None, return the
                output as string.
            name (str): name of a molecule. If `file_format` is sdf, this is the first
                line the molecule block in the sdf.
            format (str): format of the molecule, supporting: sdf, pdb, and smi.
            v3000 (bool): whether to force v3000 form if format is `sdf`
        """
        if filename is not None:
            filename = expand_path(filename)
            create_directory(filename)

        name = str(self.id) if name is None else name
        self.rdkit_mol.SetProp("_Name", name)

        if format == "sdf":
            if filename is None:
                sdf = Chem.MolToMolBlock(self.rdkit_mol, forceV3000=v3000)
                return sdf + "$$$$\n"
            else:
                return Chem.MolToMolFile(self.rdkit_mol, filename, forceV3000=v3000)
        elif format == "pdb":
            if filename is None:
                sdf = Chem.MolToPDBBlock(self.rdkit_mol)
                return sdf + "$$$$\n"
            else:
                return Chem.MolToPDBFile(self.rdkit_mol, filename)
        elif format == "smi":
            return Chem.MolToSmiles(self.rdkit_mol)
        else:
            raise ValueError(f"format {format} currently not supported")

    def draw(self, filename=None, draw_2D=True, show_atom_idx=False):
        """
        Draw using rdkit.
        """
        m = self.rdkit_mol
        if draw_2D:
            AllChem.Compute2DCoords(m)
        if show_atom_idx:
            atoms = [m.GetAtomWithIdx(i) for i in range(m.GetNumAtoms())]
            _ = [a.SetAtomMapNum(a.GetIdx() + 1) for a in atoms]

        filename = filename or "mol.png"
        filename = create_directory(filename)
        filename = expand_path(filename)
        Draw.MolToFile(m, filename)

    def pack_features(self, broken_bond=None):
        feats = dict()
        feats["charge"] = self.charge
        return feats

    def __expr__(self):
        return f"{self.id}_{self.formula}"

    def __str__(self):
        return self.__expr__()


def create_wrapper_mol_from_atoms_and_bonds(
    species, coords, bonds, charge=0, free_energy=None, identifier=None
):
    """
    Create a :class:`MoleculeWrapper` from atoms and bonds.

    Args:
        species (list of str): atom species str
        coords (2D array): positions of atoms
        bonds (list of tuple): each tuple is a bond (atom indices)
        charge (int): chare of the molecule
        free_energy (float): free energy of the molecule
        identifier (str): (unique) identifier of the molecule

    Returns:
        MoleculeWrapper instance
    """

    pymatgen_mol = pymatgen.Molecule(species, coords, charge)
    bonds = {tuple(sorted(b)): None for b in bonds}
    mol_graph = MoleculeGraph.with_edges(pymatgen_mol, bonds)

    return MoleculeWrapper(mol_graph, free_energy, identifier)


def rdkit_mol_to_wrapper_mol(m, charge=0, free_energy=None, identifier=None):
    """
    Convert an rdkit molecule to a :class:`MoleculeWrapper` molecule.

    This constructs a molecule graph from the rdkit mol and assigns the rdkit mol
    to the molecule wrapper.

    Args:
        m (Chem.Mol): rdkit molecule
        charge (int): charge of the molecule
        free_energy (float): free energy of the molecule
        identifier (str): (unique) identifier of the molecule

    Returns:
        MoleculeWrapper instance
    """

    species = [a.GetSymbol() for a in m.GetAtoms()]

    # coords = m.GetConformer().GetPositions()
    # NOTE, the above way to get coords results in segfault on linux, so we use the
    # below workaround
    conformer = m.GetConformer()
    coords = [[x for x in conformer.GetAtomPosition(i)] for i in range(m.GetNumAtoms())]

    bonds = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in m.GetBonds()]
    bonds = {tuple(sorted(b)): None for b in bonds}

    pymatgen_mol = pymatgen.Molecule(species, coords, charge)
    mol_graph = MoleculeGraph.with_edges(pymatgen_mol, bonds)

    mw = MoleculeWrapper(mol_graph, free_energy, identifier)
    mw.rdkit_mol = m

    return mw


def smiles_to_wrapper_mol(s, charge=0, free_energy=None):
    """
    Convert a smiles molecule to a :class:`MoleculeWrapper` molecule.

    3D coords are created using RDkit: embedding then MMFF force filed (or UFF force
     field).
    """

    def optimize_till_converge(method, m):
        maxiters = 200
        while True:
            error = method(m, maxIters=maxiters)
            if error == 1:
                maxiters *= 2
            else:
                return error

    try:
        # create molecules
        m = Chem.MolFromSmiles(s)
        m = Chem.AddHs(m)

        # embedding
        error = AllChem.EmbedMolecule(m, randomSeed=35)
        if error == -1:  # https://sourceforge.net/p/rdkit/mailman/message/33386856/
            AllChem.EmbedMolecule(m, randomSeed=35, useRandomCoords=True)

        # optimize, try MMFF first, if fails then UFF
        error = optimize_till_converge(AllChem.MMFFOptimizeMolecule, m)
        if error == -1:  # MMFF cannot be set up
            optimize_till_converge(AllChem.UFFOptimizeMolecule, m)

        m = rdkit_mol_to_wrapper_mol(m, charge, free_energy, s)

    # cannot convert smiles string to mol
    except ValueError as e:
        logger.warning(f"Cannot convert smiles to mol: {e}")
        m = None

    return m


def write_sdf_csv_dataset(
    molecules,
    struct_file="struct_mols.sdf",
    label_file="label_mols.csv",
    feature_file="feature_mols.yaml",
    exclude_single_atom=True,
):
    struct_file = expand_path(struct_file)
    label_file = expand_path(label_file)

    logger.info(
        "Start writing dataset to files: {} and {}".format(struct_file, label_file)
    )

    feats = []

    with open(struct_file, "w") as fx, open(label_file, "w") as fy:

        fy.write("mol_id,atomization_energy\n")

        i = 0
        for m in molecules:

            if exclude_single_atom and m.num_atoms == 1:
                logger.info("Excluding single atom molecule {}".format(m.formula))
                continue

            sdf = m.write(name=m.id + "_index-" + str(i))
            fx.write(sdf)
            fy.write("{},{:.15g}\n".format(m.id, m.atomization_free_energy))

            feats.append(m.pack_features())
            i += 1

    # write feature file
    yaml_dump(feats, feature_file)


def write_edge_label_based_on_bond(
    molecules,
    sdf_filename="mols.sdf",
    label_filename="bond_label.yaml",
    feature_filename="feature.yaml",
    exclude_single_atom=True,
):
    """
    For a molecule from SDF file, creating complete graph for atoms and label the edges
    based on whether its an actual bond or not.

    The order of the edges are (0,1), (0,2), ... , (0, N-1), (1,2), (1,3), ...,
    (N-2, N-1), where N is the number of atoms.

    Args:
        molecules (list): a sequence of MoleculeWrapper object
        sdf_filename (str): name of the output sdf file
        label_filename (str): name of the output label file
        feature_filename (str): name of the output feature file
    """

    def get_bond_label(m):
        """
        Get to know whether an edge in a complete graph is a bond.

        Returns:
            list: bool to indicate whether an edge is a bond. The edges are in the order:
                (0,1), (0,2), ..., (0,N-1), (1,2), (1,3), ..., (N, N-1), where N is the
                number of atoms.
        """
        bonds = [b for b, attr in m.bonds.items()]
        num_bonds = len(bonds)
        if num_bonds < 1:
            warnings.warn("molecular has no bonds")

        bond_label = []
        for u, v in itertools.combinations(range(m.num_atoms), 2):
            b = tuple(sorted([u, v]))
            if b in bonds:
                bond_label.append(True)
            else:
                bond_label.append(False)

        return bond_label

    labels = []
    charges = []
    sdf_filename = expand_path(sdf_filename)
    with open(sdf_filename, "w") as f:
        i = 0
        for m in molecules:

            if exclude_single_atom and m.num_atoms == 1:
                logger.info("Excluding single atom molecule {}".format(m.formula))
                continue

            sdf = m.write(name=m.id + " int_id-" + str(i))
            f.write(sdf)
            labels.append(get_bond_label(m))
            charges.append({"charge": m.charge})
            i += 1

    yaml_dump(labels, expand_path(label_filename))
    yaml_dump(charges, expand_path(feature_filename))


def fragment_mol_graph(mol_graph, bonds):
    """
    Break a bond in molecule graph and obtain the fragment(s).

    Args:
        mol_graph (MoleculeGraph): molecule graph to fragment
        bonds (list): bond indices (2-tuple)

    Returns:
        dict: with bond index (2-tuple) as key, and a list of fragments (mol_graphs)
            as values. Each list could be of size 1 or 2 and could be empty if the
            mol has no bonds.
    """
    sub_mols = {}

    for edge in bonds:
        edge = tuple(edge)
        try:
            new_mgs = mol_graph.split_molecule_subgraphs(
                [edge], allow_reverse=True, alterations=None
            )
            sub_mols[edge] = new_mgs
        except MolGraphSplitError:  # cannot split, (breaking a bond in a ring)
            new_mg = copy.deepcopy(mol_graph)
            idx1, idx2 = edge
            new_mg.break_edge(idx1, idx2, allow_reverse=True)
            sub_mols[edge] = [new_mg]
    return sub_mols


def order_two_molecules(m1, m2):
    """
    Order the molecules according to the below rules (in order):

    1. molecular mass
    2. number of atoms
    3. number of bonds
    4. alphabetical formula
    5. diameter of molecule graph, i.e. largest distance for node to node
    6. charge

    Args:
        m1, m2 : MoleculeWrapper

    Returns:
        A list of ordered molecules.
    """

    def compare(pa, pb, a, b):
        if pa < pb:
            return [a, b]
        elif pa > pb:
            return [b, a]
        else:
            return None

    def order_by_weight(a, b):
        pa = a.weight
        pb = b.weight
        return compare(pa, pb, a, b)

    def order_by_natoms(a, b):
        pa = a.num_atoms
        pb = b.num_atoms
        return compare(pa, pb, a, b)

    def order_by_nbonds(a, b):
        pa = len(a.bonds)
        pb = len(b.bonds)
        return compare(pa, pb, a, b)

    def order_by_formula(a, b):
        pa = a.formula
        pb = b.formula
        return compare(pa, pb, a, b)

    def order_by_diameter(a, b):
        try:
            pa = nx.diameter(a.graph)
        except nx.NetworkXError:
            pa = 100000000
        try:
            pb = nx.diameter(b.graph)
        except nx.NetworkXError:
            pb = 100000
        return compare(pa, pb, a, b)

    def order_by_charge(a, b):
        pa = a.charge
        pb = b.charge
        return compare(pa, pb, a, b)

    out = order_by_weight(m1, m2)
    if out is not None:
        return out
    out = order_by_natoms(m1, m2)
    if out is not None:
        return out
    out = order_by_nbonds(m1, m2)
    if out is not None:
        return out
    out = order_by_formula(m1, m2)
    if out is not None:
        return out
    out = order_by_diameter(m1, m2)
    if out is not None:
        return out

    if m1.mol_graph.isomorphic_to(m2.mol_graph):
        out = order_by_charge(m1, m2)  # e.g. H+ and H-
        if out is not None:
            return out
        else:
            return [m1, m2]  # two exactly the same molecules
    raise RuntimeError("Cannot order molecules")
