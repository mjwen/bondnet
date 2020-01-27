import copy
import logging
import warnings
import numpy as np
import networkx as nx
from collections import defaultdict
from monty.dev import requires
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
import pybel as pb
from gnn.utils import create_directory, pickle_dump, pickle_load, yaml_dump, expand_path

logger = logging.getLogger(__name__)


class BabelMolAdaptor2(BabelMolAdaptor):
    """
    Fix to BabelMolAdaptor (see FIX below):
    1. Set spin_multiplicity and charge after EndModify, otherwise, it does not take
    effect.
    2. Add and remove bonds between mol graph and obmol, since the connectivity of mol
    graph can be edited and different from the underlying pymatgen mol.
    """

    @requires(
        pb and ob,
        "BabelMolAdaptor requires openbabel to be installed with "
        "Python bindings. Please get it at http://openbabel.org.",
    )
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

    @requires(
        pb and ob,
        "BabelMolAdaptor requires openbabel to be installed with "
        "Python bindings. Please get it at http://openbabel.org.",
    )
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


# TODO should let Molecule not dependent on db_entry. The needed property from db_entry
#  should be passed in at init
class MoleculeWrapper:
    def __init__(self, db_entry, use_metal_edge_extender=True, optimized=True):
        self.db_entry = db_entry
        if optimized:
            if db_entry["state"] != "successful":
                raise UnsuccessfulEntryError
            try:
                self.mol = pymatgen.Molecule.from_dict(
                    db_entry["output"]["optimized_molecule"]
                )
            except KeyError:
                self.mol = pymatgen.Molecule.from_dict(
                    db_entry["output"]["initial_molecule"]
                )
                print(
                    "use initial_molecule for id: {}; job type:{} ".format(
                        db_entry["_id"], db_entry["output"]["job_type"]
                    )
                )
        else:
            self.mol = pymatgen.Molecule.from_dict(db_entry["input"]["initial_molecule"])

        self.use_metal_edge_extender = use_metal_edge_extender

        self._mol_graph = None
        self._ob_adaptor = None
        self._fragments = None
        self._graph_idx_to_ob_idx_map = None

    @property
    def pymatgen_mol(self):
        return self.mol

    @property
    def mol_graph(self):
        if self._mol_graph is None:
            self._mol_graph = MoleculeGraph.with_local_env_strategy(
                self.pymatgen_mol,
                OpenBabelNN(order=True),
                reorder=False,
                extend_structure=False,
            )
            if self.use_metal_edge_extender:
                self._mol_graph = metal_edge_extender(self.mol_graph)

        return self._mol_graph

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
    def id(self):
        return str(self._get_property_from_db_entry(["_id"]))

    @property
    def energy(self):
        return self._get_property_from_db_entry(["output", "final_energy"])

    @property
    def entropy(self):
        return self._get_property_from_db_entry(["output", "entropy"])

    @property
    def enthalpy(self):
        return self._get_property_from_db_entry(["output", "enthalpy"])

    @property
    def free_energy(self, T=298.0):
        if self.enthalpy is not None and self.entropy is not None:
            return (
                self.energy * 27.21139
                + self.enthalpy * 0.0433641
                - T * self.entropy * 0.0000433641
            )
        else:
            return None

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
            self._fragments = self._get_fragments()
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

    def _get_property_from_db_entry(self, keys):
        """
        Args:
            keys (list of str): the keys to go to the property in the DB. For
                example: ['output', 'entropy']
        """
        out = self.db_entry
        for k in keys:
            out = out[k]
        return out

    def _get_fragments(self):
        """
        Fragment molecule by breaking ONE bond.

        Returns:
            A dictionary with key of bond index (a tuple (idx1, idx2)), and value a list
            of the new mol_graphs (could be empty if the mol has no bonds).
        """
        sub_mols = {}
        for edge in self.graph.edges():
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


class DatabaseOperation:
    def __init__(self, entries):
        self.entries = entries

    @classmethod
    def from_query(cls, db_file="/Users/mjwen/Applications/mongo_db_access/sam_db.json"):
        """
        Query a (Sam's) database to pull all the molecules.
        """
        logger.info("Start building database from query...")

        # Create a json file that contains the credentials
        mmdb = QChemCalcDb.from_db_file(db_file, admin=True)

        # This contains all the production jobs(wb97xv/def2-tzvppd/smd(LiEC parameters)).
        # Every target_entries[i] is a dictionary of all the information for one job.
        query = {"tags.class": "smd_production"}
        entries = list(mmdb.collection.find(query))

        # to query by id: also needs to: from bson.objectid import ObjectId
        # query = {'_id': ObjectId('5d1a6e059ab9e0c05b1b2a1a')}

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

    def filter(self, keys, values):
        """
        Filter out all database entries whose value match the give value.

        Args:
            keys: list of dictionary keys
            values: a list of values to which to match

        Returns:
            list of database entries that are filtered out

        Example:
            >>> entries = ...
            >>> db = DatabaseOperation(entries)
            >>> db.filter(keys=["formula_alphabetical"], values=["H1", "H2"])
            >>> db.filter(keys=["output", 'job_type'], values=["sp"])
        """
        results = []
        for entry in self.entries:
            v = entry
            for k in keys:
                v = v[k]
            if v in values:
                results.append(entry)

        self.entries = results

    def get_job_types(self, filename="job_types.yml"):
        counts = defaultdict(list)
        for entry in self.entries:
            job_type = entry["output"]["job_type"]
            counts[job_type].append(str(entry["_id"]))
        for k, v in counts.items():
            print("number of '{}' type jobs: {}".format(k, len(v)))
        yaml_dump(counts, filename)

    def to_molecules(self, optimized=True, purify=True, sort=True):
        """
        Convert data entries to molecules.

        Args:
            optimized (bool): If True, build from optimized molecule. otherwise build
                from initial molecule.
            purify (bool): If True, return unique ones, i.e. remove ones that have higher
                free energy. Uniqueness is determined isomorphism, charge, and spin
                multiplicity. If `optimized` is False, this will be set to `False`
                internally.
            sort (bool): If True, sort molecules by their formula.

        Returns:
            A list of Molecule object
        """

        logger.info("Start converting DB entries to molecules...")

        n_unsuccessful_entry = 0
        n_unconnected_mol = 0
        n_not_unique_mol = 0

        if not optimized:
            purify = False

        mols = []
        for entry in self.entries:
            try:
                m = MoleculeWrapper(entry, optimized=optimized)
                if purify:

                    # check for connectivity
                    if not nx.is_weakly_connected(m.graph):
                        n_unconnected_mol += 1
                        continue

                    # check for isomorphism
                    idx = -1
                    for i_p_m, p_m in enumerate(mols):
                        # TODO check whether spin_multiplicity is needed
                        if (
                            m.charge == p_m.charge
                            and m.spin_multiplicity == p_m.spin_multiplicity
                            and m.mol_graph.isomorphic_to(p_m.mol_graph)
                        ):
                            n_not_unique_mol += 1
                            idx = i_p_m
                            break
                    if idx >= 0:
                        if m.free_energy < mols[idx].free_energy:
                            mols[idx] = m
                    else:
                        mols.append(m)
                else:
                    mols.append(m)
            except UnsuccessfulEntryError:
                n_unsuccessful_entry += 1
        logger.info(
            "DB entry size: {}; Unsuccessful entries {}; Isomorphic entries: {}; "
            "Unconnected molecules: {}; Entries converted to molecules: {}.".format(
                len(self.entries),
                n_unsuccessful_entry,
                n_not_unique_mol,
                n_unconnected_mol,
                len(mols),
            )
        )

        if sort:
            mols = sorted(mols, key=lambda m: m.formula)

        return mols

    def group_isomorphic(self, molecules):
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

    def write_group_isomorphic_to_file(self, molecules, filename):
        groups = self.group_isomorphic(molecules)
        filename = expand_path(filename)
        create_directory(filename)
        with open(filename, "w") as f:
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

                # entry = m.db_entry
                # conn = m.get_connectivity()
                # species = m.get_species()
                # coords = m.get_coords()
                # bond_order = m.get_bond_order()
                # print('conn', conn)
                # print('species', species)
                # print('coords', coords)
                # print('bond_order', bond_order)

                # smiles = m.write(file_format="smi")
                # print("id", m.id, "smiles", smiles, "formula", m.formula)
                #
                # filename = "images/{}_{}.svg".format(m.formula, i)
                # m.draw(filename, show_atom_idx=True)

                sdf = m.write(file_format="sdf", mol_id=m.id + " int_id-" + str(i))
                fx.write(sdf)
                fy.write(
                    "{},{},{:.15g}\n".format(m.id, m.charge, m.atomization_free_energy)
                )

                i += 1


class UnsuccessfulEntryError(Exception):
    def __init__(self):
        pass
