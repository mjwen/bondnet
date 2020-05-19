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
from atomate.qchem.database import QChemCalcDb
from gnn.database.molwrapper import MoleculeWrapper
from gnn.utils import create_directory, expand_path

logger = logging.getLogger(__name__)


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
            mol_graph, free_energy, identifier
        )

    @property
    def spin_multiplicity(self):
        return self.pymatgen_mol.spin_multiplicity

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

    def pack_features(self, broken_bond=None):
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
            mol_graph, free_energy, identifier
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
    def spin_multiplicity(self):
        return self.pymatgen_mol.spin_multiplicity

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

    def pack_features(self, broken_bond=None):
        """
        Pack the features from QChem computations into a dict.

        Args:
            broken_bond (tuple): If not `None`, include submolecule features, where
                atom level features are assigned to submolecules upon bond breaking.

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
                db_file = "/Users/mjwen/Applications/db_access/sam_db_molecules.json"
            elif db_collection == "task":
                db_file = "/Users/mjwen/Applications/db_access/sam_db.json"
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
