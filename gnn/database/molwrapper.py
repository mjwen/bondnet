"""
Molecule wrapper over pymatgen's Molecule class.
"""

import os
import copy
import logging
import warnings
import numpy as np
import itertools
import subprocess
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM
import networkx as nx
import pymatgen
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.io.babel import BabelMolAdaptor
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import openbabel as ob
from gnn.utils import create_directory, expand_path, yaml_dump

logger = logging.getLogger(__name__)


class BabelMolAdaptor2(BabelMolAdaptor):
    """
    Fix to BabelMolAdaptor (see FIX below):
    1. Set spin_multiplicity and charge after EndModify, otherwise, it does not take
    effect.
    2. Add and remove bonds between mol graph and obmol, since the connectivity of mol
    graph can be edited and different from the underlying pymatgen mol.
    """

    def __init__(self, mol):
        """
        Initializes with pymatgen Molecule or OpenBabel"s OBMol.

        Args:
            mol: pymatgen's Molecule or OpenBabel OBMol
        """
        if isinstance(mol, Molecule):
            if not mol.is_ordered:
                raise ValueError("OpenBabel Molecule only supports ordered molecules.")

            # For some reason, manually adding atoms does not seem to create
            # the correct OBMol representation to do things like force field
            # optimization. So we go through the indirect route of creating
            # an XYZ file and reading in that file.
            obmol = ob.OBMol()
            obmol.BeginModify()
            for site in mol:
                coords = [c for c in site.coords]
                atomno = site.specie.Z
                obatom = ob.OBAtom()
                obatom.thisown = 0
                obatom.SetAtomicNum(atomno)
                obatom.SetVector(*coords)
                obmol.AddAtom(obatom)
                del obatom
            obmol.ConnectTheDots()
            obmol.PerceiveBondOrders()
            obmol.Center()
            obmol.Kekulize()
            obmol.EndModify()

            # FIX 1
            obmol.SetTotalSpinMultiplicity(mol.spin_multiplicity)
            obmol.SetTotalCharge(mol.charge)

            self._obmol = obmol
        elif isinstance(mol, ob.OBMol):
            self._obmol = mol

    @staticmethod
    def from_molecule_graph(mol_graph):
        if not isinstance(mol_graph, MoleculeGraph):
            raise ValueError("not get mol graph")
        self = BabelMolAdaptor2(mol_graph.molecule)
        # FIX 2
        self._add_and_remove_bond(mol_graph)
        return self

    def add_bond(self, idx1, idx2, order=0):
        """
        Add a bond to an openbabel molecule with the specified order

        Args:
           idx1 (int): The atom index of one of the atoms participating the in bond
           idx2 (int): The atom index of the other atom participating in the bond
           order (float): Bond order of the added bond
        """
        # check whether bond exists
        for obbond in ob.OBMolBondIter(self.openbabel_mol):
            if (obbond.GetBeginAtomIdx() == idx1 and obbond.GetEndAtomIdx() == idx2) or (
                obbond.GetBeginAtomIdx() == idx2 and obbond.GetEndAtomIdx() == idx1
            ):
                raise Exception("bond exists not added")
        self.openbabel_mol.AddBond(idx1, idx2, order)

    def _add_and_remove_bond(self, mol_graph):
        """
        Add bonds in mol_graph not in obmol to obmol, and remove bonds in obmol but
        not in mol_graph.
        """

        # graph idx to ob idx map
        idx_map = dict()
        natoms = self.openbabel_mol.NumAtoms()
        for graph_idx in range(natoms):
            coords = list(mol_graph.graph.nodes[graph_idx]["coords"])
            for atom in ob.OBMolAtomIter(self.openbabel_mol):
                c = [atom.GetX(), atom.GetY(), atom.GetZ()]
                if np.allclose(c, coords):
                    idx_map[graph_idx] = atom.GetIdx()
                    break
            if graph_idx not in idx_map:
                raise Exception("atom not found in obmol.")

        # graph bonds (note that although MoleculeGraph uses multigrpah, but duplicate
        # bonds are removed when calling in MoleculeGraph.with_local_env_strategy
        graph_bonds = [
            sorted([idx_map[i], idx_map[j]]) for i, j, _ in mol_graph.graph.edges.data()
        ]

        # open babel bonds
        ob_bonds = [
            sorted([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])
            for b in ob.OBMolBondIter(self.openbabel_mol)
        ]

        # add and and remove bonds
        for bond in graph_bonds:
            if bond not in ob_bonds:
                self.add_bond(*bond, order=0)
        for bond in ob_bonds:
            if bond not in graph_bonds:
                self.remove_bond(*bond)


class BabelMolAdaptor3(BabelMolAdaptor2):
    """
    Compared to BabelMolAdaptor2, this corrects the bonds and then do other stuff like
    PerceiveBondOrders.
    NOTE: this seems create problems that OpenBabel cannot satisfy valence rule.

    Fix to BabelMolAdaptor (see FIX below):
    1. Set spin_multiplicity and charge after EndModify, otherwise, it does not take
    effect.
    2. Add and remove bonds between mol graph and obmol, since the connectivity of mol
    graph can be edited and different from the underlying pymatgen mol.
    """

    def __init__(self, mol_graph):
        """
        Initializes with pymatgen Molecule or OpenBabel"s OBMol.

        """
        mol = mol_graph.molecule
        if isinstance(mol, Molecule):
            if not mol.is_ordered:
                raise ValueError("OpenBabel Molecule only supports ordered molecules.")

            # For some reason, manually adding atoms does not seem to create
            # the correct OBMol representation to do things like force field
            # optimization. So we go through the indirect route of creating
            # an XYZ file and reading in that file.
            obmol = ob.OBMol()
            obmol.BeginModify()
            for site in mol:
                coords = [c for c in site.coords]
                atomno = site.specie.Z
                obatom = ob.OBAtom()
                obatom.thisown = 0
                obatom.SetAtomicNum(atomno)
                obatom.SetVector(*coords)
                obmol.AddAtom(obatom)
                del obatom
            obmol.ConnectTheDots()

            self._obmol = obmol

            # FIX 2
            self._add_and_remove_bond(mol_graph)

            obmol.PerceiveBondOrders()
            obmol.Center()
            obmol.Kekulize()
            obmol.EndModify()

            # FIX 1
            obmol.SetTotalSpinMultiplicity(mol.spin_multiplicity)
            obmol.SetTotalCharge(mol.charge)

        elif isinstance(mol, ob.OBMol):
            self._obmol = mol

    @staticmethod
    def from_molecule_graph(mol_graph):
        if not isinstance(mol_graph, MoleculeGraph):
            raise ValueError("not get mol graph")
        return BabelMolAdaptor3(mol_graph)


class MoleculeWrapper:
    """
    A wrapper arould pymatgen Molecule, MoleculeGraph, BabelAdaptor... to make it
    easier to use molecules.

    This is a base class, and typically you do not use this directly but instead using
    the derived class, e.g. MoleculeWrapperMolBuilder, MoleculeWrapperTaskCollection,
    MoleculeWrapperFromAtomsAndBonds.
    """

    def __init__(self):
        # should be set upon creation
        self.id = None
        self.free_energy = None
        self.pymatgen_mol = None
        self.mol_graph = None

        # set when corresponding method is called
        self._ob_adaptor = None
        self._fragments = None
        self._isomorphic_bonds = None
        self._graph_idx_to_ob_idx_map = None
        self._ob_idx_to_graph_idx_map = None

    @property
    def ob_adaptor(self):
        if self._ob_adaptor is None:
            self._ob_adaptor = BabelMolAdaptor2.from_molecule_graph(self.mol_graph)
        return self._ob_adaptor

    @property
    def ob_mol(self):
        return self.ob_adaptor.openbabel_mol

    @property
    def pybel_mol(self):
        return self.ob_adaptor.pybel_mol

    @property
    def rdkit_mol(self):
        sdf = self.write(file_format="sdf")
        return Chem.MolFromMolBlock(sdf)

    @property
    def atomization_free_energy(self):
        charge0_atom_energy = {
            "H": -13.899716296436546,
            "Li": -203.8840240968338,
            "C": -1028.6825101424483,
            "O": -2040.4807693439561,
            "F": -2714.237000742088,
            "P": -9283.226337212582,
        }

        e = self.free_energy
        for spec, num in self.composition_dict.items():
            e -= charge0_atom_energy[spec] * num
        return e

    @property
    def charge(self):
        return self.pymatgen_mol.charge

    @property
    def spin_multiplicity(self):
        return self.pymatgen_mol.spin_multiplicity

    @property
    def graph(self):
        return self.mol_graph.graph

    @property
    def atoms(self):
        """
        Sorted atoms of in the molecule.
        
        Returns:
            list: each component is a dict of atom attributes.
        """
        nodes = self.graph.nodes.data()
        return [v for k, v in sorted(nodes, key=lambda pair: pair[0])]

    @property
    def bonds(self):
        """
        Returns:
            dict: with bond index (a tuple of atom indices) as the key and and bond
                attributes as the value.

        """
        return {tuple(sorted([i, j])): attr for i, j, attr in self.graph.edges.data()}

    @property
    def species(self):
        return [v["specie"] for v in self.atoms]

    @property
    def coords(self):
        return np.asarray([v["coords"] for v in self.atoms])

    @property
    def formula(self):
        f = self.pymatgen_mol.composition.alphabetical_formula
        return f.replace(" ", "")

    @property
    def composition_dict(self):
        d = self.pymatgen_mol.composition.as_dict()
        return {k: int(v) for k, v in d.items()}

    @property
    def weight(self):
        return self.pymatgen_mol.composition.weight

    @property
    def fragments(self):
        if self._fragments is None:
            self._fragments = self.get_fragments()
        return self._fragments

    @property
    def isomorphic_bonds(self):
        r"""
        Find isomorphic bonds. For example, given the molecule

            H1---C0---H2
                /  \
              O3---O4

        bond 0 and bond 1 are isomorphically identical, and bond 2 and bond 3 are also
        isomorphically identical.

        Returns:
            list (ist): each inner list contains the indices (tuple) of bonds that
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

    @property
    def graph_idx_to_ob_idx_map(self):
        if self._graph_idx_to_ob_idx_map is None:
            self._graph_idx_to_ob_idx_map = dict()
            for graph_idx in range(len(self.atoms)):
                coords = list(self.graph.nodes[graph_idx]["coords"])
                for atom in ob.OBMolAtomIter(self.ob_mol):
                    c = [atom.GetX(), atom.GetY(), atom.GetZ()]
                    if np.allclose(c, coords):
                        self._graph_idx_to_ob_idx_map[graph_idx] = atom.GetIdx()
                        break
                if graph_idx not in self._graph_idx_to_ob_idx_map:
                    raise Exception("atom not found.")
        return self._graph_idx_to_ob_idx_map

    def graph_bond_idx_to_ob_bond_idx(self, bond):
        idx0 = self.graph_idx_to_ob_idx_map[bond[0]]
        idx1 = self.graph_idx_to_ob_idx_map[bond[1]]
        return (idx0, idx1)

    @property
    def ob_idx_to_graph_idx_map(self):
        if self._ob_idx_to_graph_idx_map is None:
            self._ob_idx_to_graph_idx_map = dict()
            for k, v in self.graph_idx_to_ob_idx_map.items():
                self._ob_idx_to_graph_idx_map[v] = k
        return self._ob_idx_to_graph_idx_map

    def ob_bond_idx_to_graph_bond_idx(self, bond):
        idx0 = self.ob_idx_to_graph_idx_map[bond[0]]
        idx1 = self.ob_idx_to_graph_idx_map[bond[1]]
        return (idx0, idx1)

    def make_picklable(self):
        self._ob_adaptor = None

    def get_sdf_bond_indices(self, sdf=None):
        sdf = sdf or self.write(file_format="sdf")
        lines = sdf.split("\n")
        split_3 = lines[3].split()
        natoms = int(split_3[0])
        nbonds = int(split_3[1])
        bonds = []
        for line in lines[4 + natoms : 4 + natoms + nbonds]:
            bonds.append(tuple(sorted([int(i) for i in line.split()[:2]])))
        return bonds

    def get_fragments(self, bonds=None):
        """
        Fragment molecule by breaking ONE bond.

        Args: a list of tuple as bond indices.

        Returns:
            A dictionary with key of bond index (a tuple (idx1, idx2)), and value a list
            of the new mol_graphs (could be empty if the mol has no bonds).
        """
        sub_mols = {}

        if bonds is None:
            bonds = [b for b, _ in self.bonds.items()]
        for edge in bonds:
            try:
                new_mgs = self.mol_graph.split_molecule_subgraphs(
                    [edge], allow_reverse=True, alterations=None
                )
                sub_mols[edge] = new_mgs
            except MolGraphSplitError:  # cannot split, i.e. open ring
                new_mg = copy.deepcopy(self.mol_graph)
                idx1, idx2 = edge
                new_mg.break_edge(idx1, idx2, allow_reverse=True)
                sub_mols[edge] = [new_mg]
        return sub_mols

    def subgraph_atom_mapping(self, bond):
        """
        Break a bond in a molecule and find the atoms mapping in the two subgraphs.

        Returns:
            tuple of list: each list contains the atoms in one subgraph.
        """

        original = copy.deepcopy(self.mol_graph)
        original.break_edge(bond[0], bond[1], allow_reverse=True)

        # A -> B breaking
        if nx.is_weakly_connected(original.graph):
            mapping = list(range(len(self.atoms)))
            return mapping, mapping
        # A -> B + C breaking
        else:
            components = nx.weakly_connected_components(original.graph)
            nodes = [original.graph.subgraph(c).nodes for c in components]
            mapping = tuple([sorted(list(n)) for n in nodes])
            if len(mapping) != 2:
                raise Exception("Mol not split into two parts")
            return mapping

    def write(self, filename=None, file_format="sdf", message=None):
        if filename is not None:
            filename = expand_path(filename)
            create_directory(filename)
        message = str(self.id) if message is None else message
        self.ob_mol.SetTitle(message)
        return self.pybel_mol.write(file_format, filename, overwrite=True)

    def draw(self, filename=None, draw_2D=True, show_atom_idx=False):
        """
        Draw using rdkit.
        """
        sdf = self.write(file_format="sdf")

        m = Chem.MolFromMolBlock(sdf)
        if m is None:
            warnings.warn("cannot draw mol")
            return
        if draw_2D:
            AllChem.Compute2DCoords(m)
        if show_atom_idx:
            atoms = [m.GetAtomWithIdx(i) for i in range(m.GetNumAtoms())]
            _ = [a.SetAtomMapNum(a.GetIdx() + 1) for a in atoms]

        filename = filename or "mol.png"
        filename = create_directory(filename)
        Draw.MolToFile(m, filename)

    def draw2(self, filename=None, draw_2D=True, show_atom_idx=False):
        """
        Draw using pybel.
        """

        filename = filename or "mol.png"
        filename = create_directory(filename)
        if draw_2D:
            usecoords = False
        else:
            usecoords = True
        self.pybel_mol.draw(show=False, filename=filename, usecoords=usecoords)

    def draw3(self, filename=None, draw_2D=True, show_atom_idx=False):
        """
        Draw using obabel cmdline tool.
        """

        sdf = self.write(file_format="sdf")

        # remove sdf M attributes except the ones in except_M
        # except_M = ["RAD"]
        except_M = []
        new_sdf = sdf.split("\n")
        sdf = []
        for line in new_sdf:
            if "M" in line:
                keep = False
                for ecpt in except_M:
                    if ecpt in line:
                        keep = True
                        break
                if not keep:
                    continue
            sdf.append(line)
        sdf = "\n".join(sdf)

        # write the sdf to a file
        sdf_name = "graph.sdf"
        with open(sdf_name, "w") as f:
            f.write(sdf)

        # use obable to write svg file
        svg_name = "graph.svg"
        command = ["obabel", "graph.sdf", "-O", svg_name, "-xa", "-xd"]
        if show_atom_idx:
            command += ["-xi"]
        subprocess.run(command)

        # convert format
        filename = filename or "mol.svg"
        filename = create_directory(filename)
        path, extension = os.path.splitext(filename)
        if extension == ".svg":
            subprocess.run(["cp", svg_name, filename])
        else:
            try:
                drawing = svg2rlg(svg_name)
                if extension == ".pdf":
                    renderPDF.drawToFile(drawing, filename)
                elif extension == ".png":
                    renderPM.drawToFile(drawing, filename, fmt="PNG")
                else:
                    raise Exception(
                        "file format `{}` not support. Supported are pdf and png.".format(
                            extension
                        )
                    )
            except AttributeError:
                print("Cannot convert to {} file for {}".format(extension), self.id)

        # remove temporary files
        subprocess.run(["rm", sdf_name])
        subprocess.run(["rm", svg_name])


class MoleculeWrapperFromAtomsAndBonds(MoleculeWrapper):
    """
    A molecule wrapper class that creates molecules by giving species, coords,
    and bonds. It will not have many properties, e.g. free_energy.
    """

    def __init__(self, species, coords, charge, bonds, mol_id=None, free_energy=None):

        super(MoleculeWrapperFromAtomsAndBonds, self).__init__()

        self.id = mol_id
        self.pymatgen_mol = pymatgen.Molecule(species, coords, charge)

        bonds = {tuple(sorted(b)): None for b in bonds}
        self.mol_graph = MoleculeGraph.with_edges(self.pymatgen_mol, bonds)
        self.free_energy = free_energy

    def pack_features(self, use_obabel_idx=True, broken_bond=None):
        feats = dict()
        feats["charge"] = self.charge

        return feats


def rdkit_mol_to_wrapper_mol(m, charge=0, mol_id=None):

    species = [m.GetAtomWithIdx(i).GetSymbol() for i in range(m.GetNumAtoms())]

    # coords = m.GetConformer().GetPositions()
    # NOTE, the above way to get coords results in segfault on linux, so we use the below
    # workaround
    conformer = m.GetConformer()
    coords = [[x for x in conformer.GetAtomPosition(i)] for i in range(m.GetNumAtoms())]

    bonds = []
    for i in range(m.GetNumBonds()):
        b = m.GetBondWithIdx(i)
        bonds.append([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])

    return MoleculeWrapperFromAtomsAndBonds(species, coords, charge, bonds, mol_id)


def ob_mol_to_wrapper_mol(m, charge=0, mol_id=None):

    species = [a.GetAtomicNum() for a in ob.OBMolAtomIter(m)]
    coords = [[a.GetX(), a.GetY(), a.GetZ()] for a in ob.OBMolAtomIter(m)]
    bonds = [
        sorted([b.GetBeginAtomIdx(), b.GetEndAtomIdx()]) for b in ob.OBMolBondIter(m)
    ]
    bonds = np.asarray(bonds) - 1  # convert to zero index

    return MoleculeWrapperFromAtomsAndBonds(species, coords, charge, bonds, mol_id)


def smiles_to_wrapper_mol(s, charge=0):
    """Convert a smiles molecule to a :class:`MoleculeWrapper` molecule.

       3D coords are created using RDkit: embedding then MMFF force filed (or UFF force
       field).
    """

    # babel way to do it
    # m = pybel.readstring("smi", s)
    # m.addh()
    # m.make3D()
    # m.localopt()
    # m = ob_mol_to_wrapper_mol(m.OBMol, charge=0, mol_id=s)

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

        m = rdkit_mol_to_wrapper_mol(m, charge=charge, mol_id=s)

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

            if exclude_single_atom and len(m.atoms) == 1:
                logger.info("Excluding single atom molecule {}".format(m.formula))
                continue

            # The same pybel mol will write different sdf file when it is called
            # the first time and other times. We create a new one by setting
            # `_ob_adaptor` to None here so that it will write the correct one.
            m._ob_adaptor = None
            sdf = m.write(file_format="sdf", message=m.id + " int_id-" + str(i))
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

        num_atoms = len(m.atoms)
        bond_label = []
        for u, v in itertools.combinations(range(num_atoms), 2):
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

            if exclude_single_atom and len(m.atoms) == 1:
                logger.info("Excluding single atom molecule {}".format(m.formula))
                continue

            m._ob_adaptor = None
            sdf = m.write(file_format="sdf", message=m.id + " int_id-" + str(i))
            f.write(sdf)
            labels.append(get_bond_label(m))
            charges.append({"charge": m.charge})
            i += 1

    yaml_dump(labels, expand_path(label_filename))
    yaml_dump(charges, expand_path(feature_filename))
