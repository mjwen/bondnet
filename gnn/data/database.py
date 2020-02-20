"""
Convert Sam's database entries to molecules.
"""
import os
import copy
import logging
import warnings
import numpy as np
import itertools
import subprocess
from collections import defaultdict
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM
import networkx as nx
from atomate.qchem.database import QChemCalcDb
import pymatgen
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.analysis.fragmenter import metal_edge_extender
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
        graph_bonds = []
        for i, j, _ in mol_graph.graph.edges.data():
            graph_bonds.append(sorted([idx_map[i], idx_map[j]]))

        # open babel bonds
        ob_bonds = []
        for bond in ob.OBMolBondIter(self.openbabel_mol):
            ob_bonds.append(sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))

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


class BaseMoleculeWrapper:
    def __init__(self, db_entry):
        self.id = None
        self.free_energy = None
        self.pymatgen_mol = None
        self.mol_graph = None
        self._ob_adaptor = None
        self._fragments = None
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
        return self.nx_graph()

    def nx_graph(self):
        return self.mol_graph.graph

    @property
    def atoms(self):
        return self.graph.nodes.data()

    @property
    def bonds(self):
        return self.graph.edges.data()

    @property
    def species(self):
        return self._get_node_attr("specie")

    @property
    def coords(self):
        return self._get_node_attr("coords")

    @property
    def bond_order(self):
        return self._get_edge_attr("weight")

    @property
    def formula(self):
        f = self.pymatgen_mol.composition.alphabetical_formula
        return f.replace(" ", "")

    @property
    def composition_dict(self):
        return self.pymatgen_mol.composition.as_dict()

    @property
    def weight(self):
        return self.pymatgen_mol.composition.weight

    @property
    def fragments(self):
        if self._fragments is None:
            self._fragments = self.get_fragments()
        return self._fragments

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

    def _get_node_attr(self, attr):
        return [a for _, a in self.graph.nodes.data(attr)]

    def _get_edge_attr(self, attr):
        return [a for _, _, a in self.graph.edges.data(attr)]

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
            bonds = self.graph.edges()
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
            mapping = tuple([list(sorted(list(n))) for n in nodes])
            if len(mapping) != 2:
                raise Exception("Mole not split into two parts")
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


class MoleculeWrapperTaskCollection(BaseMoleculeWrapper):
    def __init__(self, db_entry, use_metal_edge_extender=True, optimized=True):

        super(MoleculeWrapperTaskCollection, self).__init__(db_entry)

        self.id = str(db_entry["_id"])

        self.free_energy = self._get_free_energy(db_entry, self.id)

        if optimized:
            if db_entry["state"] != "successful":
                raise UnsuccessfulEntryError
            try:
                self.pymatgen_mol = pymatgen.Molecule.from_dict(
                    db_entry["output"]["optimized_molecule"]
                )
            except KeyError:
                self.pymatgen_mol = pymatgen.Molecule.from_dict(
                    db_entry["output"]["initial_molecule"]
                )
                print(
                    "use initial_molecule for id: {}; job type:{} ".format(
                        db_entry["_id"], db_entry["output"]["job_type"]
                    )
                )
        else:
            self.pymatgen_mol = pymatgen.Molecule.from_dict(
                db_entry["input"]["initial_molecule"]
            )

        self.mol_graph = MoleculeGraph.with_local_env_strategy(
            self.pymatgen_mol,
            OpenBabelNN(order=True),
            reorder=False,
            extend_structure=False,
        )
        if use_metal_edge_extender:
            self.mol_graph = metal_edge_extender(self.mol_graph)

    def pack_features(self, use_obabel_idx=True):
        feats = dict()

        # molecule level
        feats["id"] = self.id
        feats["free_energy"] = self.free_energy
        feats["atomization_free_energy"] = self.atomization_free_energy
        feats["charge"] = self.charge
        feats["spin_multiplicity"] = self.spin_multiplicity

        return feats

    @staticmethod
    def _get_free_energy(entry, mol_id, T=300):

        try:
            energy = entry["output"]["energy"]
        except KeyError as e:
            print(e, "energy", mol_id)

        try:
            entropy = entry["output"]["entropy"]
        except KeyError as e:
            print(e, "entropy", mol_id)
            raise UnsuccessfulEntryError

        try:
            enthalpy = entry["output"]["enthalpy"]
        except KeyError as e:
            print(e, "enthalpy", mol_id)
            raise UnsuccessfulEntryError

        return energy * 27.21139 + enthalpy * 0.0433641 - T * entropy * 0.0000433641


class MoleculeWrapperMolBuilder(BaseMoleculeWrapper):
    def __init__(self, db_entry):
        super(MoleculeWrapperMolBuilder, self).__init__(db_entry)

        self.id = str(db_entry["_id"])

        try:
            self.free_energy = db_entry["free_energy"]
        except KeyError as e:
            print(self.__class__.__name__, e, "free energy", self.id)
            raise UnsuccessfulEntryError

        try:
            self.resp = db_entry["resp"]
        except KeyError as e:
            print(self.__class__.__name__, e, "resp", self.id)
            raise UnsuccessfulEntryError

        try:
            mulliken = db_entry["mulliken"]
            if len(np.asarray(mulliken).shape) == 1:  # partial spin is 0
                self.mulliken = [i for i in mulliken]
                self.atom_spin = [0 for _ in mulliken]
            else:
                self.mulliken = [i[0] for i in mulliken]
                self.atom_spin = [i[1] for i in mulliken]
        except KeyError as e:
            print(self.__class__.__name__, e, "mulliken", self.id)
            raise UnsuccessfulEntryError

        try:
            self.pymatgen_mol = pymatgen.Molecule.from_dict(db_entry["molecule"])
        except KeyError as e:
            print(self.__class__.__name__, e, "molecule", self.id)
            raise UnsuccessfulEntryError

        try:
            self.mol_graph = MoleculeGraph.from_dict(db_entry["mol_graph"])
        except KeyError as e:
            # NOTE it would be better that we can get MoleculeGraph from database
            if db_entry["nsites"] == 1:  # single atom molecule
                self.mol_graph = MoleculeGraph.with_local_env_strategy(
                    self.pymatgen_mol,
                    OpenBabelNN(order=True),
                    reorder=False,
                    extend_structure=False,
                )
            else:
                print(self.__class__.__name__, e, "free energy", self.id)
                raise UnsuccessfulEntryError

        # critic
        try:
            self.critic = db_entry["critic"]
        except KeyError as e:
            print(self.__class__.__name__, e, "critic", self.id)
            raise UnsuccessfulEntryError

    def convert_to_babel_mol_graph(self, use_metal_edge_extender=True):
        self._ob_adaptor = None
        self.mol_graph = MoleculeGraph.with_local_env_strategy(
            self.pymatgen_mol,
            OpenBabelNN(order=True),
            reorder=False,
            extend_structure=False,
        )
        if use_metal_edge_extender:
            self.mol_graph = metal_edge_extender(self.mol_graph)

    def convert_to_critic_mol_graph(self):
        self._ob_adaptor = None
        bonds = dict()
        try:
            for key, val in self.critic["bonding"].items():
                idx = val["atom_ids"]
                idx = tuple([int(i) - 1 for i in idx])
                bonds[idx] = None
        except KeyError as e:
            print(self.__class__.__name__, e, "critic bonding", self.id)
            raise UnsuccessfulEntryError
        self.mol_graph = MoleculeGraph.with_edges(self.pymatgen_mol, bonds)

    def pack_features(self, use_obabel_idx=True):
        feats = dict()

        # molecule level
        feats["id"] = self.id
        feats["free_energy"] = self.free_energy
        feats["atomization_free_energy"] = self.atomization_free_energy
        feats["charge"] = self.charge
        feats["spin_multiplicity"] = self.spin_multiplicity

        # atom level
        resp = [i for i in self.resp]
        mulliken = [i for i in self.mulliken]
        atom_spin = [i for i in self.atom_spin]

        if use_obabel_idx:
            resp_old = copy.deepcopy(resp)
            mulliken_old = copy.deepcopy(mulliken)
            asm_old = copy.deepcopy(atom_spin)
            for graph_idx in range(len(self.atoms)):
                ob_idx = self.graph_idx_to_ob_idx_map[graph_idx]
                # -1 because ob index starts from 1
                resp[ob_idx - 1] = resp_old[graph_idx]
                mulliken[ob_idx - 1] = mulliken_old[graph_idx]
                atom_spin[ob_idx - 1] = asm_old[graph_idx]

        feats["resp"] = resp
        feats["mulliken"] = mulliken
        feats["atom_spin"] = atom_spin

        return feats


class DatabaseOperation:
    @staticmethod
    def query_db_entries(db_collection="mol_builder", db_file=None, num_entries=None):
        """
        Query a (Sam's) database to pull all the molecules form molecule builder.

        Args:
            db_collection (str): which database to query. Optionals are `mol_builder`
                and `smd`.
            db_file (str): a json file storing the info of the database.
            num_entries (int): the number of entries to query, if `None`, get all.

        Returns:
            A list of db entries.
        """

        logger.info("Start querying database...")

        if db_file is None:
            if db_collection == "mol_builder":
                db_file = (
                    "/Users/mjwen/Applications/mongo_db_access/sam_db_molecules.json"
                )
            elif db_collection == "task":
                db_file = "/Users/mjwen/Applications/mongo_db_access/sam_db.json"
            else:
                raise Exception("Unrecognized db_collection = {}".format(db_collection))

        mmdb = QChemCalcDb.from_db_file(db_file, admin=True)

        if db_collection == "mol_builder":
            if num_entries is None:
                entries = mmdb.collection.find()
            else:
                entries = mmdb.collection.find().limit(num_entries)
        elif db_collection == "task":
            query = {"tags.class": "smd_production"}
            if num_entries is None:
                entries = mmdb.collection.find(query)
            else:
                entries = mmdb.collection.find(query).limit(num_entries)
        else:
            raise Exception("Unrecognized db_collection = {}".format(db_collection))

        entries = list(entries)
        logger.info(
            "Finish fetching {} entries of database from query...".format(len(entries))
        )

        return entries

    @staticmethod
    def to_molecules(entries, db_collection="mol_builder", sort=True):
        """
        Convert data entries to molecules.

        Args:
            db_collection (str): which database to query. Optionals are `mol_builder`
                and `smd`.
            sort (bool): If True, sort molecules by their formula.

        Returns:
            A list of MoleculeWrapper object
        """

        logger.info("Start converting DB entries to molecules...")

        if db_collection == "mol_builder":
            MW = MoleculeWrapperMolBuilder
        elif db_collection == "smd":
            MW = MoleculeWrapperSMD
        else:
            raise Exception("Unrecognized db_collection = {}".format(db_collection))

        unsuccessful = 0
        mols = []
        for i, entry in enumerate(entries):
            if i // 100 == 0:
                logger.info(
                    "Converted {}/{} entries to molecules.".format(i, len(entries))
                )
            try:
                m = MW(entry)
                mols.append(m)
            except UnsuccessfulEntryError:
                unsuccessful += 1

        logger.info(
            "Total entries: {}, unsuccessful: {}, successful: {} to molecules.".format(
                len(entries), unsuccessful, len(entries) - unsuccessful
            )
        )

        if sort:
            mols = sorted(mols, key=lambda m: m.formula)

        return mols

    @staticmethod
    def filter_molecules(molecules, connectivity=True, isomorphism=True):
        """
        Filter out some molecules.

        Args:
            molecules (list of MoleculeWrapper): molecules
            connectivity (bool): whether to filter on connectivity
            isomorphism (bool): if `True`, filter on `charge`, `spin`, and `isomorphism`.
                The one with the lowest free energy will remain.

        Returns:
            A list of MoleculeWrapper objects.
        """
        n_unconnected_mol = 0
        n_not_unique_mol = 0

        filtered = []
        for i, m in enumerate(molecules):

            # check on connectivity
            if connectivity and not nx.is_weakly_connected(m.graph):
                n_unconnected_mol += 1
                continue

            # check for isomorphism
            if isomorphism:
                idx = -1
                for i_p_m, p_m in enumerate(filtered):
                    if (
                        m.charge == p_m.charge
                        and m.spin_multiplicity == p_m.spin_multiplicity
                        and m.mol_graph.isomorphic_to(p_m.mol_graph)
                    ):
                        n_not_unique_mol += 1
                        idx = i_p_m
                        break
                if idx >= 0:
                    if m.free_energy < filtered[idx].free_energy:
                        filtered[idx] = m
                else:
                    filtered.append(m)

        print(
            "Num molecules: {}; unconnected: {}; isomorphic: {}; remaining: {}".format(
                len(molecules), n_unconnected_mol, n_not_unique_mol, len(filtered)
            )
        )

        return filtered

    @staticmethod
    def write_group_isomorphic_to_file(molecules, filename):
        def group_isomorphic(molecules):
            """
            Group molecules
            Args:
                molecules: a list of Molecules.

            Returns:
                A list of list, with inner list of isomorphic molecules.
            """
            groups = []
            for m in molecules:
                find_iso = False
                for g in groups:
                    iso_m = g[0]
                    if m.mol_graph.isomorphic_to(iso_m.mol_graph):
                        g.append(m)
                        find_iso = True
                        break
                if not find_iso:
                    groups.append([m])
            return groups

        groups = group_isomorphic(molecules)

        # statistics or charges of mols
        charges = defaultdict(int)
        for m in molecules:
            charges[m.charge] += 1

        # statistics of isomorphic mols
        sizes = defaultdict(int)
        for g in groups:
            sizes[len(g)] += 1

        # statistics of charge combinations
        charge_combinations = defaultdict(int)
        for g in groups:
            chg = [m.charge for m in g]
            for ij in itertools.combinations(chg, 2):
                ij = tuple(sorted(ij))
                charge_combinations[ij] += 1

        filename = expand_path(filename)
        create_directory(filename)
        with open(filename, "w") as f:
            f.write("Number of molecules: {}\n\n".format(len(molecules)))
            f.write("Molecule charge state statistics.\n")
            f.write("# charge state     number of molecules:\n")
            for k, v in charges.items():
                f.write("{}    {}\n".format(k, v))

            f.write("Number of isomorphic groups: {}\n\n".format(len(groups)))
            f.write(
                "Molecule isomorphic group size statistics. (i.e. the number of "
                "isomorphic molecules that have a specific number of charge state\n"
            )
            f.write("# size     number of molecules:\n")
            for k, v in sizes.items():
                f.write("{}    {}\n".format(k, v))

            f.write("# charge combinations     number:\n")
            for k, v in charge_combinations.items():
                f.write("{}    {}\n".format(k, v))

            for g in groups:
                for m in g:
                    f.write("{}_{}_{}    ".format(m.formula, m.id, m.charge))
                f.write("\n")

    @staticmethod
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


class UnsuccessfulEntryError(Exception):
    def __init__(self):
        pass
