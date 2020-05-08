"""
Query the electrolyte database to get molecules.
"""

import logging
import numpy as np
import itertools
from collections import defaultdict
import networkx as nx
import pymatgen
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.analysis.fragmenter import metal_edge_extender
from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor
from atomate.qchem.database import QChemCalcDb
from gnn.database.molwrapper import MoleculeWrapper
from gnn.utils import create_directory, expand_path
import openbabel as ob

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


class MoleculeWrapperTaskCollection(MoleculeWrapper):
    def __init__(self, db_entry, use_metal_edge_extender=True, optimized=True):

        # id
        identifier = str(db_entry["_id"])

        # pymatgen mol
        if optimized:
            if db_entry["state"] != "successful":
                raise UnsuccessfulEntryError
            try:
                pymatgen_mol = pymatgen.Molecule.from_dict(
                    db_entry["output"]["optimized_molecule"]
                )
            except KeyError:
                pymatgen_mol = pymatgen.Molecule.from_dict(
                    db_entry["output"]["initial_molecule"]
                )
                print(
                    "use initial_molecule for id: {}; job type:{} ".format(
                        db_entry["_id"], db_entry["output"]["job_type"]
                    )
                )
        else:
            pymatgen_mol = pymatgen.Molecule.from_dict(
                db_entry["input"]["initial_molecule"]
            )

        # mol graph
        mol_graph = MoleculeGraph.with_local_env_strategy(
            pymatgen_mol, OpenBabelNN(order=True), reorder=False, extend_structure=False,
        )
        if use_metal_edge_extender:
            mol_graph = metal_edge_extender(self.mol_graph)

        # free energy
        free_energy = self._get_free_energy(db_entry, self.id, self.formula)

        super(MoleculeWrapperTaskCollection, self).__init__(
            pymatgen_mol, mol_graph, free_energy, identifier
        )

    def create_ob_mol(self):
        ob_adaptor = BabelMolAdaptor2.from_molecule_graph(self.mol_graph)
        return ob_adaptor.openbabel_mol

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
    def _get_free_energy(entry, mol_id, formula, T=300):

        try:
            energy = entry["output"]["energy"]
        except KeyError as e:
            print(e, "energy", mol_id, formula)

        try:
            entropy = entry["output"]["entropy"]
        except KeyError as e:
            print(e, "entropy", mol_id, formula)
            raise UnsuccessfulEntryError

        try:
            enthalpy = entry["output"]["enthalpy"]
        except KeyError as e:
            print(e, "enthalpy", mol_id, formula)
            raise UnsuccessfulEntryError

        return energy * 27.21139 + enthalpy * 0.0433641 - T * entropy * 0.0000433641


class MoleculeWrapperMolBuilder(MoleculeWrapper):
    def __init__(self, db_entry):

        # id
        identifier = str(db_entry["_id"])

        # pymatgen mol
        try:
            pymatgen_mol = pymatgen.Molecule.from_dict(db_entry["molecule"])
        except KeyError as e:
            print(self.__class__.__name__, e, "molecule", identifier)
            raise UnsuccessfulEntryError
        formula = pymatgen_mol.composition.alphabetical_formula.replace(" ", "")

        # mol graph
        try:
            mol_graph = MoleculeGraph.from_dict(db_entry["mol_graph"])
        except KeyError as e:
            # NOTE it would be better that we can get MoleculeGraph from database
            if db_entry["nsites"] == 1:  # single atom molecule
                mol_graph = MoleculeGraph.with_local_env_strategy(
                    pymatgen_mol,
                    OpenBabelNN(order=True),
                    reorder=False,
                    extend_structure=False,
                )
            else:
                print(
                    "conversion failed",
                    self.__class__.__name__,
                    e,
                    "free energy",
                    identifier,
                    formula,
                )
                raise UnsuccessfulEntryError

        # free energy
        try:
            free_energy = db_entry["free_energy"]
        except KeyError as e:
            print(self.__class__.__name__, e, "free energy", identifier, formula)
            raise UnsuccessfulEntryError

        super(MoleculeWrapperMolBuilder, self).__init__(
            pymatgen_mol, mol_graph, free_energy, identifier
        )

        # other properties
        try:
            self.resp = db_entry["resp"]
        except KeyError as e:
            print(self.__class__.__name__, e, "resp", self.id, self.formula)
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
            print(self.__class__.__name__, e, "mulliken", self.id, self.formula)
            raise UnsuccessfulEntryError

        # critic
        try:
            self.critic = db_entry["critic"]
        except KeyError as e:
            if db_entry["nsites"] != 1:  # not single atom molecule
                print(self.__class__.__name__, e, "critic", self.id, self.formula)
                raise UnsuccessfulEntryError

    def create_ob_mol(self):
        ob_adaptor = BabelMolAdaptor2.from_molecule_graph(self.mol_graph)
        return ob_adaptor.openbabel_mol

    def convert_to_babel_mol_graph(self, use_metal_edge_extender=True):
        self.mol_graph = MoleculeGraph.with_local_env_strategy(
            self.pymatgen_mol,
            OpenBabelNN(order=True),
            reorder=False,
            extend_structure=False,
        )
        if use_metal_edge_extender:
            self.mol_graph = metal_edge_extender(self.mol_graph)

    def convert_to_critic_mol_graph(self):
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

    def pack_features(self, use_obabel_idx=True, broken_bond=None):
        """
        Pack the features from QChem computations into a dict.

        Args:
            use_obabel_idx (bool): If `True`, atom level features (e.g. `resp`) will
                use babel atom index, otherwise, use graph atom index.
            broken_bond (tuple): If not `None`, include submolecule features, where
                atom level features are assigned to submolecules upon bond breaking.
                Note `broken_bond` should be given in graph atom index, regardless of
                the value of `use_obabel_idx`.

        Returns:
            dict: features
        """
        feats = dict()

        # molecule level
        feats["id"] = self.id
        feats["free_energy"] = self.free_energy
        feats["atomization_free_energy"] = self.atomization_free_energy
        feats["charge"] = self.charge
        feats["spin_multiplicity"] = self.spin_multiplicity

        # submolecules level (upon bond breaking)
        if broken_bond is not None:
            mappings = self.subgraph_atom_mapping(broken_bond)
            sub_resp = []
            sub_mulliken = []
            sub_atom_spin = []
            for mp in mappings:
                sub_resp.append([self.resp[i] for i in mp])
                sub_mulliken.append([self.mulliken[i] for i in mp])
                sub_atom_spin.append([self.atom_spin[i] for i in mp])
            feats["abs_resp_diff"] = abs(sum(sub_resp[0]) - sum(sub_resp[1]))
            feats["abs_mulliken_diff"] = abs(sum(sub_mulliken[0]) - sum(sub_mulliken[1]))
            feats["abs_atom_spin_diff"] = abs(
                sum(sub_atom_spin[0]) - sum(sub_atom_spin[1])
            )

        # atom level
        resp = [i for i in self.resp]
        mulliken = [i for i in self.mulliken]
        atom_spin = [i for i in self.atom_spin]
        if use_obabel_idx:
            for graph_idx in range(len(self.atoms)):
                ob_idx = self.graph_idx_to_ob_idx_map[graph_idx]
                # -1 because ob index starts from 1
                resp[ob_idx - 1] = self.resp[graph_idx]
                mulliken[ob_idx - 1] = self.mulliken[graph_idx]
                atom_spin[ob_idx - 1] = self.atom_spin[graph_idx]

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
                and `task`.
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
                and `task`.
            sort (bool): If True, sort molecules by their formula.

        Returns:
            A list of MoleculeWrapper object
        """

        logger.info("Start converting DB entries to molecules...")

        if db_collection == "mol_builder":
            MW = MoleculeWrapperMolBuilder
        elif db_collection == "task":
            MW = MoleculeWrapperTaskCollection
        else:
            raise Exception("Unrecognized db_collection = {}".format(db_collection))

        unsuccessful = 0
        mols = []
        for i, entry in enumerate(entries):
            if i % 100 == 0:
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

        logger.info(
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


class UnsuccessfulEntryError(Exception):
    def __init__(self):
        pass
