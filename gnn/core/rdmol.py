import warnings
import numpy as np
import logging
from collections import defaultdict
from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor
from rdkit import Chem
from rdkit.Chem import BondType, AllChem
from rdkit.Geometry import Point3D
import openbabel as ob
from gnn.utils import expand_path

logger = logging.getLogger(__name__)


def smiles_to_rdkit_mol(s, add_H=True):
    """
    Convert a smiles string to rdkit molecule.

    3D coords are created using RDkit: embedding then MMFF force filed (or UFF force
     field).

    Args:
        s (str): smiles of the molecule
        add_H (bool): whether to add H to the molecule

    Returns:
        rdkit mol
    """
    m = Chem.MolFromSmiles(s)
    if m is None:
        raise RdkitMolCreationError(f"smiles: {s}")

    if add_H:
        m = Chem.AddHs(m)

    try:
        m = generate_3D_coords(m)
    except GenerateCoordsError as e:
        raise RdkitMolCreationError(f"smiles: {e}")

    m.SetProp("_Name", s)

    return m


def smarts_to_rdkit_mol(s):
    """
    Convert a smarts string to rdkit molecule.

    3D coords are created using RDkit: embedding then MMFF force filed (or UFF force
     field).

    Args:
        s (str): smarts of the molecule

    Returns:
        rdkit mol
    """
    m = Chem.MolFromSmarts(s)
    if m is None:
        raise RdkitMolCreationError(f"smiles: {s}")

    Chem.SanitizeMol(m)

    try:
        m = generate_3D_coords(m)
    except GenerateCoordsError as e:
        raise RdkitMolCreationError(f"smiles: {e}")

    m.SetProp("_Name", s)

    return m


def inchi_to_rdkit_mol(s):
    """
    Convert a inchi string to rdkit molecule.

    3D coords are created using RDkit: embedding then MMFF force filed (or UFF force
     field).

    Args:
        s (str): inchi of the molecule

    Returns:
        rdkit mol
    """
    m = Chem.MolFromInchi(s, sanitize=True, removeHs=False)
    if m is None:
        raise RdkitMolCreationError(f"inchi: {s}")
    m = Chem.AddHs(m)

    try:
        m = generate_3D_coords(m)
    except GenerateCoordsError as e:
        raise RdkitMolCreationError(f"smiles: {e}")

    m.SetProp("_Name", s)

    return m


def read_rdkit_mols_from_file(filename, format="sdf"):
    """
    Read molecules given in a file, supported format include `smiles`, `inchi`, `sdf`,
    and `pdb`.

    Args:
        filename (str): name of the file
        format (str): format of the molecules in the file

    Returns:
        (list): rdkit molecules
    """

    def from_smiles(filename):
        smiles = []
        with open(expand_path(filename), "r") as f:
            for line in f:
                smiles.append(line.strip())

        molecules = []
        for s in smiles:
            try:
                m = smiles_to_rdkit_mol(s)
            except RdkitMolCreationError:
                m = None
            molecules.append(m)

        return molecules

    def from_inchi(filename):
        inchi = []
        with open(expand_path(filename), "r") as f:
            for line in f:
                inchi.append(line.strip())

        molecules = []
        for s in inchi:
            try:
                m = inchi_to_rdkit_mol(s)
            except RdkitMolCreationError:
                m = None
            molecules.append(m)

        return molecules

    def from_sdf(filename):
        supp = Chem.SDMolSupplier(filename, sanitize=True, removeHs=False)
        molecules = [m for m in supp]

        return molecules

    def from_pdb(filename):
        with open(expand_path(filename), "r") as f:
            pdbs = f.read()
        pdbs = [s.strip() for s in pdbs.split("$$$$\n")]
        pdbs = [s for s in pdbs if s != ""]
        molecules = [Chem.MolFromPDBBlock(p, sanitize=True, removeHs=False) for p in pdbs]

        return molecules

    if format == "smiles":
        return from_smiles(filename)
    elif format == "inchi":
        return from_inchi(filename)
    elif format == "sdf":
        return from_sdf(filename)
    elif format == "pdb":
        return from_pdb(filename)
    else:
        raise ValueError(
            f"Not supported molecule format `{format}`; choose one of "
            f"`smiles`, `inchi`, `sdf` or `pdb` instead."
        )


def create_rdkit_mol(
    species, coords, bond_types, formal_charge=None, name=None, force_sanitize=True
):
    """
    Create a rdkit mol from scratch.

    Followed: https://sourceforge.net/p/rdkit/mailman/message/36474923/

    Args:
        species (list): species str of each molecule
        coords (2D array): positions of atoms
        bond_types (dict): with bond indices (2 tuple) as key and bond type
            (e.g. Chem.rdchem.BondType.DOUBLE) as value
        formal_charge (list): formal charge of each atom
        name (str): name of the molecule
        force_sanitize (bool): whether to force the sanitization of molecule.
            If `True` and the sanitization fails, it generally throw an error
            and then stops. If `False`, will try to sanitize first, but if it fails,
            will proceed smoothly giving a warning message.

    Returns:
        rdkit Chem.Mol
    """

    m = Chem.Mol()
    edm = Chem.EditableMol(m)
    conformer = Chem.Conformer(len(species))

    for i, (s, c) in enumerate(zip(species, coords)):
        atom = Chem.Atom(s)
        atom.SetNoImplicit(True)
        if formal_charge is not None:
            cg = formal_charge[i]
            if cg is not None:
                atom.SetFormalCharge(cg)
        atom_idx = edm.AddAtom(atom)
        conformer.SetAtomPosition(atom_idx, Point3D(*c))

    for b, t in bond_types.items():
        edm.AddBond(b[0], b[1], t)

    m = edm.GetMol()
    if force_sanitize:
        Chem.SanitizeMol(m)
    else:
        try:
            Chem.SanitizeMol(m)
        except Exception as e:
            warnings.warn(f"Cannot sanitize molecule {name}, because {str(e)}")
    m.AddConformer(conformer, assignId=False)

    m.SetProp("_Name", str(name))

    return m


def create_rdkit_mol_from_mol_graph(
    mol_graph, name=None, force_sanitize=False, metals={"Li": 1, "Mg": 2}
):
    """
    Create a rdkit molecule from molecule graph, with bond type perceived by babel.
    Done in the below steps:

    1. create a babel mol without metal atoms.
    2. perceive bond order (conducted by BabelMolAdaptor)
    3. adjust formal charge of metal atoms so as not to violate valence rule
    4. create rdkit mol based on species, coords, bonds, and formal charge

    Args:
        mol_graph (pymatgen MoleculeGraph): molecule graph
        name (str): name of the molecule
        force_sanitize (bool): whether to force sanitization of the rdkit mol
        metals dict: with metal atom (str) as key and the number of valence electrons
            as key.

    Returns:
        m: rdkit Chem.Mol
        bond_types (dict): bond types assigned to the created rdkit mol
    """

    pymatgen_mol = mol_graph.molecule
    species = [str(s) for s in pymatgen_mol.species]
    coords = pymatgen_mol.cart_coords
    bonds = [tuple(sorted([i, j])) for i, j, attr in mol_graph.graph.edges.data()]

    # create babel mol without metals
    pmg_mol_no_metals = remove_metals(pymatgen_mol)
    adaptor = BabelMolAdaptor(pmg_mol_no_metals)
    ob_mol = adaptor.openbabel_mol

    # get babel bond order of mol without metals
    ob_bond_order = {}
    for bd in ob.OBMolBondIter(ob_mol):
        k = tuple(sorted([bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()]))
        v = bd.GetBondOrder()
        ob_bond_order[k] = v

    # create bond type
    atom_idx_mapping = pymatgen_2_babel_atom_idx_map(pymatgen_mol, ob_mol)
    bond_types = {}

    for bd in bonds:
        try:
            ob_bond = [atom_idx_mapping[a] for a in bd]

            # atom not in ob mol
            if None in ob_bond:
                raise KeyError
            # atom in ob mol
            else:
                ob_bond = tuple(sorted(ob_bond))
                v = ob_bond_order[ob_bond]
                if v == 0:
                    tp = BondType.UNSPECIFIED
                elif v == 1:
                    tp = BondType.SINGLE
                elif v == 2:
                    tp = BondType.DOUBLE
                elif v == 3:
                    tp = BondType.TRIPLE
                elif v == 5:
                    tp = BondType.AROMATIC
                else:
                    raise RuntimeError(f"Got unexpected babel bond order: {v}")

        except KeyError:
            atom1_spec, atom2_spec = [species[a] for a in bd]

            if atom1_spec in metals and atom2_spec in metals:
                raise RuntimeError("Got a bond between two metal atoms")

            # bond involves one and only one metal atom (atom not in ob mol case above)
            elif atom1_spec in metals or atom2_spec in metals:
                tp = Chem.rdchem.BondType.DATIVE

                # Dative bonds have the special characteristic that they do not affect
                # the valence on the start atom, but do affect the end atom.
                # Here we adjust the atom ordering in the bond for dative bond to make
                # metal the end atom.
                if atom1_spec in metals:
                    bd = tuple(reversed(bd))

            # bond not found by babel (atom in ob mol)
            else:
                tp = Chem.rdchem.BondType.UNSPECIFIED

        bond_types[bd] = tp

    # a metal atom can form multiple dative bond (e.g. bidentate LiEC), for such cases
    # we need to adjust the their formal charge so as not to violate valence rule
    formal_charge = adjust_formal_charge(species, bonds, metals)

    m = create_rdkit_mol(species, coords, bond_types, formal_charge, name, force_sanitize)

    return m, bond_types


def adjust_formal_charge(species, bonds, metals):
    """
    Adjust formal charge of metal atoms.

    Args:
        species (list): species string of atoms
        bonds (list): 2-tuple index of bonds
        metals (dict): intial formal charge of models

    Returns:
        list: formal charge of atoms. None for non metal atoms.
    """
    # initialize formal charge first so that atom does not form any bond has its formal
    # charge set
    formal_charge = [metals[s] if s in metals else None for s in species]

    # atom_idx: idx of atoms in the molecule
    # num_bonds: number of bonds the atom forms
    atom_idx, num_bonds = np.unique(bonds, return_counts=True)
    for i, ct in zip(atom_idx, num_bonds):
        s = species[i]
        if s in metals:
            formal_charge[i] = int(formal_charge[i] - ct)

    return formal_charge


def generate_3D_coords(m):
    """
    Generate 3D coords for an rdkit molecule.

    This is done by embedding and then optimizing using MMFF force filed (or UFF force
    field).

    Args:
        m (Chem.Mol): rdkit mol.

    Return:
        Chem.Mol: rdkit mol with updated coords
    """

    def optimize_till_converge(method, m):
        maxiters = 200
        while True:
            error = method(m, maxIters=maxiters)
            if error == 1:
                maxiters *= 2
            else:
                return error

    # embedding
    error = AllChem.EmbedMolecule(m, randomSeed=35)
    if error == -1:  # https://sourceforge.net/p/rdkit/mailman/message/33386856/
        AllChem.EmbedMolecule(m, randomSeed=35, useRandomCoords=True)
    if error == -1:
        raise GenerateCoordsError()

    # optimize, try MMFF first, if fails then UFF
    error = optimize_till_converge(AllChem.MMFFOptimizeMolecule, m)
    if error == -1:  # MMFF cannot be set up
        optimize_till_converge(AllChem.UFFOptimizeMolecule, m)

    return m


def pymatgen_2_babel_atom_idx_map(pmg_mol, ob_mol):
    """
    Create an atom index mapping between pymatgen mol and openbabel mol.

    This does not require pymatgen mol and ob mol has the same number of atoms.
    But ob_mol can have smaller number of atoms.

    Args:
        pmg_mol (pymatgen.Molecule): pymatgen molecule
        ob_mol (ob.Mol): openbabel molecule

    Returns:
        dict: with atom index in pymatgen mol as key and atom index in babel mol as
            value. Value is `None` if there is not corresponding atom in babel.
    """

    pmg_coords = pmg_mol.cart_coords
    ob_coords = [[a.GetX(), a.GetY(), a.GetZ()] for a in ob.OBMolAtomIter(ob_mol)]
    ob_index = [a.GetIdx() for a in ob.OBMolAtomIter(ob_mol)]

    mapping = {i: None for i in range(len(pmg_coords))}

    for idx, oc in zip(ob_index, ob_coords):
        for i, gc in enumerate(pmg_coords):
            if np.allclose(oc, gc):
                mapping[i] = idx
                break
        else:
            raise RuntimeError("Cannot create atom index mapping pymatgen and ob mols")

    return mapping


def remove_metals(mol, metals={"Li": 1, "Mg": 2}):
    """
    Check whether metals are in a pymatgen molecule. If yes, create a new Molecule
    with metals removed.

    Args:
        mol (pymatgen mol): molecule
        metals (dict): with metal specie are key and charge as value

    Returns:
        pymatgen mol
    """
    species = [str(s) for s in mol.species]

    if set(species).intersection(set(metals.keys())):
        charge = mol.charge

        species = []
        coords = []
        properties = defaultdict(list)
        for site in mol:
            s = str(site.specie)
            if s in metals:
                charge -= metals[s]
            else:
                species.append(s)
                coords.append(site.coords)
                for k, v in site.properties:
                    properties[k].append(v)

        # do not provide spin_multiplicity, since we remove an atom
        mol = Molecule(species, coords, charge, site_properties=properties)

    return mol


def fragment_rdkit_mol(m, bond):
    """
    Break a bond in rdkit molecule and obtain the fragment(s).

    Args:
        m (Chem.Mol): rdkit molecule to fragment
        bond (tuple): bond index (2-tuple)

    Returns:
        frags (list): fragments (rdkit molecules) by breaking the bond.
            Could be of size 1 or 2, depending on the number of fragments.
    """

    def create_rdkit_mol_from_fragment(m, name, bond_atom):
        """
        Convert a rdkit mol fragment to a rdkit mol with cleans.

        The formal charge is set to `None` for the atom in the broken bond (it will
        then be assigned when sanitize the mol) and untouched for other atoms
        (having the value as in the parent molecule).
        """
        species = [a.GetSymbol() for a in m.GetAtoms()]

        # coords = m.GetConformer().GetPositions()
        # NOTE, the above way to get coords results in segfault on linux, so we use the
        # below workaround
        conformer = m.GetConformer()
        coords = [
            [x for x in conformer.GetAtomPosition(i)] for i in range(m.GetNumAtoms())
        ]

        # should not sort (b.GetBeginAtomIdx(), b.GetEndAtomIdx()), because dative bond
        # needs to have the metal as the end atom
        bond_types = {
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx()): b.GetBondType()
            for b in m.GetBonds()
        }

        # adjust format charge
        formal_charge = [a.GetFormalCharge() for a in m.GetAtoms()]
        formal_charge[bond_atom] = None

        new_m = create_rdkit_mol(
            species, coords, bond_types, formal_charge, name, force_sanitize=True
        )

        return new_m

    edm = Chem.EditableMol(m)
    edm.RemoveBond(*bond)
    m1 = edm.GetMol()
    atom_mapping = []
    frags = Chem.GetMolFrags(
        m1, asMols=True, sanitizeFrags=True, fragsMolAtomMapping=atom_mapping
    )

    # Although we passed sanitizeFrags=True to Chem.GetMolFrags. Some properties are
    # still incorrect:
    # 1. features of the returned frags. e.g. TotalDegree() of an atom of the broken
    #    bond still has the value before bond breaking.
    # 2. rdkit converts N(=O)=O to [N+](=O)O-] when sanitizing the mol, see
    #    http://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
    #    So we need to copy the formal charge from the parent mol to the fragments,
    #    otherwise, sanitizing will result in error for the fragments.
    # To address these, we create frags from scratch using info from the frags.

    new_frags = []
    for i, fg in enumerate(frags):
        for frag_atom_idx, parent_atom_idx in enumerate(atom_mapping[i]):
            if parent_atom_idx in bond:
                bond_atom = frag_atom_idx
                break
        else:
            # this should never happen
            raise RuntimeError("Cannot find bond atom in fragments")

        name = f"{m.GetProp('_Name')}_frag{i}"
        fg = create_rdkit_mol_from_fragment(fg, name, bond_atom)
        new_frags.append(fg)

    return new_frags


def smarts_atom_mapping(s):
    """
    Get the atom mapping specified in smarts.

    Args:
        s (str): smarts of the molecule

    Returns:
        list: atom mapping specified in smarts

    >>> s = '[C:1]([N:3]=[C:2]=[N:6][N:5]=[O:4])([H:7])([H:8])[H:9]'
    >>> smarts_atom_mapping(s)
    >>> [0, 2, 1, 5, 4, 3, 6, 7, 8]
    """
    m = Chem.MolFromSmarts(s)
    if m is None:
        raise RdkitMolCreationError(f"smarts: {s}")

    ind_map = []
    for atom in m.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num != 0:
            ind_map.append(map_num - 1)
        else:
            ind_map.append(None)

    return ind_map


class GenerateCoordsError(Exception):
    def __init__(self, msg=None):
        self.msg = msg
        super(GenerateCoordsError, self).__init__(msg)

    def __repr__(self):
        return f"cannot generate 3D coords, {self.msg}"


class RdkitMolCreationError(Exception):
    def __init__(self, msg=None):
        self.msg = msg
        super(RdkitMolCreationError, self).__init__(msg)

    def __repr__(self):
        return f"cannot create rdkit mol, {self.msg}"
