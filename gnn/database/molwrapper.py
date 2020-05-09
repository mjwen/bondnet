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
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import openbabel as ob
import pybel
from gnn.utils import create_directory, expand_path, yaml_dump

logger = logging.getLogger(__name__)


class BabelMolAdaptor2(BabelMolAdaptor):
    """
    Fix to BabelMolAdaptor (see FIX below):
    1. Add and remove bonds between mol graph and obmol, since the connectivity of mol
    graph can be edited and different from the underlying pymatgen mol.
    """

    @staticmethod
    def from_molecule_graph(mol_graph):
        if not isinstance(mol_graph, MoleculeGraph):
            raise ValueError("not get mol graph")
        self = BabelMolAdaptor2(mol_graph.molecule)
        # FIX 1
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

        idx_map = graph2ob_atom_idx_map(mol_graph, self.openbabel_mol)

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
    1. Add and remove bonds between mol graph and obmol, since the connectivity of mol
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

            # FIX 1
            self._add_and_remove_bond(mol_graph)

            obmol.PerceiveBondOrders()
            obmol.SetTotalSpinMultiplicity(mol.spin_multiplicity)
            obmol.SetTotalCharge(mol.charge)
            obmol.Center()
            obmol.Kekulize()
            obmol.EndModify()

        elif isinstance(mol, ob.OBMol):
            self._obmol = mol

    @staticmethod
    def from_molecule_graph(mol_graph):
        if not isinstance(mol_graph, MoleculeGraph):
            raise ValueError("not get mol graph")
        return BabelMolAdaptor2(mol_graph)


class MoleculeWrapper:
    """
    A wrapper arould pymatgen Molecule, MoleculeGraph, BabelAdaptor... to make it
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
        self._ob_mol = None
        self._fragments = None
        self._isomorphic_bonds = None
        self._graph_to_ob_atom_idx_map = None
        self._ob_to_graph_atom_idx_map = None

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
        f = self.pymatgen_mol.composition.alphabetical_formula
        return f.replace(" ", "")

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
    def atoms(self):
        """
        Sorted atoms of in the molecule.
        
        Returns:
            list: each component is a dict of atom attributes.
        """
        nodes = self.graph.nodes.data()
        return [v for k, v in sorted(nodes, key=lambda pair: pair[0])]

    @property
    def species(self):
        """
        Species of atoms. Order is the same as self.atoms.
        Returns:
            list: Species string.
        """
        return [v["specie"] for v in self.atoms]

    @property
    def coords(self):
        """
        Returns:
            2D array: of shape (N, 3) where N is the number of atoms.
        """
        return np.asarray([v["coords"] for v in self.atoms])

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
    def ob_mol(self):
        """
        Returns:
            OpenBabel molecule

        """
        if self._ob_mol is None:
            self._ob_mol = self._create_ob_mol()
        return self._ob_mol

    @ob_mol.setter
    def ob_mol(self, m):
        self._ob_mol = m

    def delete_ob_mol(self):
        """
        This is needed in two places:
        1. ob mol is not pickable, so if we want to pickle this class, we need to
            delete it.
        2. when writing out sdf files, calling the `write` function a second time will
            write a different sdf than the first time. So we may want to delete it and
            create a new ob mol each time we write sdf.
        """
        self._ob_mol = None

    @property
    def pybel_mol(self):
        return pybel.Molecule(self._ob_mol)

    @property
    def rdkit_mol(self):
        sdf = self.write(file_format="sdf")
        return Chem.MolFromMolBlock(sdf)

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
    def graph_to_ob_atom_idx_map(self):
        """
        Returns:
            dict: atom index in graph as key in ob mol as value
        """
        if self._graph_to_ob_atom_idx_map is None:
            self._graph_to_ob_atom_idx_map = graph2ob_atom_idx_map(
                self.mol_graph, self.ob_mol
            )
        return self._graph_to_ob_atom_idx_map

    @property
    def ob_to_graph_atom_idx_map(self):
        """
        Returns:
            dict: atom index in ob mol as key in mol graph
        """
        if self._ob_to_graph_atom_idx_map is None:
            self._ob_to_graph_atom_idx_map = {
                v: k for k, v in self.graph_to_ob_atom_idx_map.items()
            }
        return self._ob_to_graph_atom_idx_map

    def graph_to_ob_bond_idx_map(self, bond):
        idx0 = self.graph_to_ob_atom_idx_map[bond[0]]
        idx1 = self.graph_to_ob_atom_idx_map[bond[1]]
        return idx0, idx1

    def ob_to_graph_bond_idx_map(self, bond):
        idx0 = self.ob_to_graph_atom_idx_map[bond[0]]
        idx1 = self.ob_to_graph_atom_idx_map[bond[1]]
        return idx0, idx1

    def get_sdf_bond_indices(self, sdf=None):
        """
        Get the indices of bonds as specified in the sdf file.

        Returns:
            list of tuple: each tuple specifies a bond.
        """
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
            of the mol_graphs of the fragments (of size 1 or 2 and could be empty if the
            mol has no
            bonds).
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
        """Write a molecule out.

        Args:
            filename (str): name of the file to write the output. If None, return the
                output as string.
            message (str): message to attach to a molecule. If `file_format` is sdf,
                this is the first line the molecule block in the sdf.
        """
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

    def pack_features(self, use_obabel_idx=True, broken_bond=None):
        feats = dict()
        feats["charge"] = self.charge
        return feats

    def _create_ob_mol(self):
        ob_adaptor = BabelMolAdaptor2.from_molecule_graph(self.mol_graph)
        return ob_adaptor.openbabel_mol


class MoleculeWrapperFromAtomsAndBonds(MoleculeWrapper):
    """
    A molecule wrapper class that creates molecules by giving species, coords,
    and bonds.
    """

    def __init__(self, species, coords, charge, bonds, mol_id=None, free_energy=None):

        pymatgen_mol = pymatgen.Molecule(species, coords, charge)
        bonds = {tuple(sorted(b)): None for b in bonds}
        mol_graph = MoleculeGraph.with_edges(pymatgen_mol, bonds)

        super(MoleculeWrapperFromAtomsAndBonds, self).__init__(
            mol_graph, free_energy, mol_id
        )


def rdkit_mol_to_wrapper_mol(m, charge=0, free_energy=None, mol_id=None):

    # use V3000 to minimize conversion error in V2000, see (although it is not sure how
    # relevant it is)
    # https://depth-first.com/articles/2012/01/11/on-the-futility-of-extending-the-molfile-format/
    sdf = Chem.MolToMolBlock(m, forceV3000=True)
    pb_mol = pybel.readstring("sdf", sdf)
    ob_mol = pb_mol.OBMol

    return ob_mol_to_wrapper_mol(ob_mol, charge, free_energy, mol_id)


def ob_mol_to_wrapper_mol(m, charge=0, free_energy=None, mol_id=None):
    """
    Convert an openbabel mol to wrapper mol.

    The created wrapper mol cannot be pickled, because the passed ob mol `m` cannot be
    pickled.
    """

    species = [a.GetAtomicNum() for a in ob.OBMolAtomIter(m)]
    coords = [[a.GetX(), a.GetY(), a.GetZ()] for a in ob.OBMolAtomIter(m)]
    bonds = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in ob.OBMolBondIter(m)]
    bonds = np.asarray(bonds) - 1  # convert to zero index

    pymatgen_mol = pymatgen.Molecule(species, coords, charge)
    bonds = {tuple(sorted(b)): None for b in bonds}
    mol_graph = MoleculeGraph.with_edges(pymatgen_mol, bonds)

    mw = MoleculeWrapper(mol_graph, free_energy, mol_id)
    mw.ob_mol = m

    return mw


def smiles_to_wrapper_mol(s, charge=0, free_energy=None):
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
            # the first time and other times. We create a new one here so that it will
            # write the correct one.
            m.delete_ob_mol()
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

            # The same pybel mol will write different sdf file when it is called
            # the first time and other times. We create a new one here so that it will
            # write the correct one.
            m.delete_ob_mol()
            sdf = m.write(file_format="sdf", message=m.id + " int_id-" + str(i))
            f.write(sdf)
            labels.append(get_bond_label(m))
            charges.append({"charge": m.charge})
            i += 1

    yaml_dump(labels, expand_path(label_filename))
    yaml_dump(charges, expand_path(feature_filename))


def graph2ob_atom_idx_map(mol_graph, ob_mol):
    """
    Create an atom index mapping between mol graph and ob mol.

    This is implemented by comparing coords.

    Returns:
        dict: with atom index in mol graph as key and atom index in ob mol as value.
    """
    mapping = dict()

    nodes = mol_graph.graph.nodes.data()
    graph_coords = [v["coords"] for k, v in sorted(nodes, key=lambda pair: pair[0])]

    ob_coords = [[a.GetX(), a.GetY(), a.GetZ()] for a in ob.OBMolAtomIter(ob_mol)]
    ob_index = [a.GetIdx() for a in ob.OBMolAtomIter(ob_mol)]

    for i, gc in enumerate(graph_coords):
        for idx, oc in zip(ob_index, ob_coords):
            if np.allclose(oc, gc):
                mapping[i] = idx
                break
        else:
            raise RuntimeError("Cannot create atom index mapping between ")

    return mapping
