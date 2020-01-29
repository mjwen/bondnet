"""
Database using molecules directly from Sam's molecule builder.
"""

import copy
import logging
import warnings
import numpy as np
import itertools
from collections import defaultdict
import networkx as nx
from atomate.qchem.database import QChemCalcDb
import pymatgen
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.analysis.fragmenter import metal_edge_extender
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import openbabel as ob
from gnn.utils import create_directory, pickle_dump, pickle_load, expand_path
from gnn.data.database import BabelMolAdaptor2 as BabelMolAdaptor

logger = logging.getLogger(__name__)


class MoleculeWrapper:
    def __init__(self, db_entry):
        # property from db entry
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
            self.mulliken = db_entry["mulliken"]
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

        # other properties
        self._ob_adaptor = None
        self._fragments = None
        self._graph_idx_to_ob_idx_map = None

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
    def ob_adaptor(self):
        if self._ob_adaptor is None:
            self._ob_adaptor = BabelMolAdaptor.from_molecule_graph(self.mol_graph)
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

    def graph_bond_idx_to_ob_bond_idx(self, bond):
        """
        Convert mol_graph bond indices to babel bond indices.
        """
        idx0 = self.graph_idx_to_ob_idx_map[bond[0]]
        idx1 = self.graph_idx_to_ob_idx_map[bond[1]]
        return (idx0, idx1)

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

    def get_connectivity(self):
        conn = []
        for u, v in self.graph.edges():
            conn.append([u, v])
        return conn

    def get_species(self):
        return self._get_node_attr("specie")

    def get_coords(self):
        return self._get_node_attr("coords")

    def get_bond_order(self):
        return self._get_edge_attr("weight")

    def make_picklable(self):
        self._ob_adaptor = None

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

    def write(self, filename=None, file_format="sdf", message=None):
        if filename is not None:
            filename = expand_path(filename)
            create_directory(filename)
        message = str(self.id) if message is None else message
        self.ob_mol.SetTitle(message)
        return self.pybel_mol.write(file_format, filename, overwrite=True)

    def draw(self, filename=None, draw_2D=True, show_atom_idx=False):
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
        filename = filename or "mol.svg"
        filename = create_directory(filename)
        Draw.MolToFile(m, filename)

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
        if len(np.asarray(self.mulliken).shape) == 1:  # partial spin is 0
            mulliken = [i for i in self.mulliken]
            atom_spin_multiplicity = [0 for i in self.mulliken]
        else:
            mulliken = [i[0] for i in self.mulliken]
            atom_spin_multiplicity = [i[1] for i in self.mulliken]

        if use_obabel_idx:
            resp_old = copy.deepcopy(resp)
            mulliken_old = copy.deepcopy(mulliken)
            asm_old = copy.deepcopy(atom_spin_multiplicity)
            for graph_idx in range(len(self.atoms)):
                ob_idx = self.graph_idx_to_ob_idx_map[graph_idx]
                # -1 because ob index starts from 1
                resp[ob_idx - 1] = resp_old[graph_idx]
                mulliken[ob_idx - 1] = mulliken_old[graph_idx]
                atom_spin_multiplicity[ob_idx - 1] = asm_old[graph_idx]

        feats["resp"] = resp
        feats["mulliken"] = mulliken
        feats["atom_spin_multiplicity"] = atom_spin_multiplicity

        return feats


class DatabaseOperation:
    def __init__(self, entries):
        self.entries = entries

    @classmethod
    def from_query(
        cls,
        db_file="/Users/mjwen/Applications/mongo_db_access/sam_db_molecules.json",
        num_entries=None,
    ):
        """
        Query a (Sam's) database to pull all the molecules form molecule builder.
        """
        logger.info("Start building database from query...")

        mmdb = QChemCalcDb.from_db_file(db_file, admin=True)

        if num_entries is None:
            entries = mmdb.collection.find()
        else:
            entries = mmdb.collection.find().limit(num_entries)
        entries = list(entries)

        logger.info(
            "Finish loading {} entries of database from query...".format(len(entries))
        )

        return cls(entries)

    @classmethod
    def from_file(cls, filename="database.pkl"):
        """
        Recover a dumped database entries.
        """
        entries = pickle_load(filename)
        logger.info(
            "{} entries loaded from database file: {}".format(len(entries), filename)
        )
        return cls(entries)

    def to_file(self, filename="database.pkl", size=None):
        """
        Dump a list of database molecule entries to disk.
        This is purely for efficiency stuff, since the next time we want to query the
        database, we can simply load the dumped one instead of query the actual database.
        """
        logger.info("Start writing database to file: {}".format(filename))
        size = size or -1
        entries = self.entries[:size]
        pickle_dump(entries, filename)

    def to_molecules(self, sort=True):
        """
        Convert data entries to molecules.

        Args:
            sort (bool): If True, sort molecules by their formula.

        Returns:
            A list of MoleculeWrapper object
        """

        logger.info("Start converting DB entries to molecules...")

        unsuccessful = 0

        mols = []
        for i, entry in enumerate(self.entries):
            if i // 100 == 0:
                logger.info(
                    "Converted {}/{} entries to molecules.".format(i, len(self.entries))
                )
            try:
                m = MoleculeWrapper(entry)
                mols.append(m)
            except UnsuccessfulEntryError:
                unsuccessful += 1

        logger.info(
            "Total entries: {}, unsuccessful: {}, successful: {} to molecules.".format(
                len(self.entries), unsuccessful, len(self.entries) - unsuccessful
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
        structure_name="electrolyte_struct.sdf",
        label_name="electrolyte_label.csv",
        exclude_single_atom=True,
    ):
        structure_name = expand_path(structure_name)
        label_name = expand_path(label_name)

        logger.info(
            "Start writing dataset to files: {} and {}".format(structure_name, label_name)
        )

        with open(structure_name, "w") as fx, open(label_name, "w") as fy:

            fy.write("mol_id,charge,atomization_energy\n")

            i = 0
            for m in molecules:

                if exclude_single_atom and len(m.atoms) == 1:
                    logger.info("Excluding single atom molecule {}".format(m.formula))
                    continue

                sdf = m.write(file_format="sdf", message=m.id + " int_id-" + str(i))
                fx.write(sdf)
                fy.write(
                    "{},{},{:.15g}\n".format(m.id, m.charge, m.atomization_free_energy)
                )

                i += 1


class UnsuccessfulEntryError(Exception):
    def __init__(self):
        pass
