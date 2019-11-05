import copy
import itertools
import logging
import warnings
import json
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from atomate.qchem.database import QChemCalcDb
import pymatgen
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.analysis.fragmenter import metal_edge_extender
from pymatgen.io.babel import BabelMolAdaptor
import openbabel as ob
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from gnn.data.utils import (
    create_directory,
    pickle_dump,
    pickle_load,
    yaml_dump,
    expand_path,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BabelMolAdaptor2(BabelMolAdaptor):
    def add_bond(self, idx1, idx2, order):
        """
        Add a bond to an openbabel molecule with the specified order

        Args:
           idx1 (int): The atom index of one of the atoms participating the in bond
           idx2 (int): The atom index of the other atom participating in the bond
           order (float): Bond order of the added bond
        """
        # TODO more clever way to handle this, see the add_edge of `MoleculeGraph`
        # check whether bond exists
        for obbond in ob.OBMolBondIter(self._obmol):
            if (obbond.GetBeginAtomIdx() == idx1 and obbond.GetEndAtomIdx() == idx2) or (
                obbond.GetBeginAtomIdx() == idx2 and obbond.GetEndAtomIdx() == idx1
            ):
                raise Exception("bond exists")

        self._obmol.AddBond(idx1, idx2, order)

    @classmethod
    def from_molecule_graph(cls, mol_graph):
        if not isinstance(mol_graph, MoleculeGraph):
            raise ValueError("not get mol graph")
        self = cls(mol_graph.molecule)
        self._add_missing_bond(mol_graph)

        return self

    def _add_missing_bond(self, mol_graph):
        def is_ob_bonds(coords1, coords2):

            for i, atom in enumerate(ob.OBMolAtomDFSIter(self._obmol)):
                cdi = [atom.GetX(), atom.GetY(), atom.GetZ()]
                if not np.allclose(cdi, coords1):
                    continue
                for neighbor in ob.OBAtomAtomIter(atom):
                    cdn = [neighbor.GetX(), neighbor.GetY(), neighbor.GetZ()]
                    if np.allclose(cdn, coords2):
                        return True
            return False

        def find_ob_index(coords):
            for atom in ob.OBMolAtomDFSIter(self._obmol):
                c = [atom.GetX(), atom.GetY(), atom.GetZ()]
                if np.allclose(c, coords):
                    return atom.GetIdx()
            raise Exception("atom not found.")  # should never get here

        for i, j, attr in mol_graph.graph.edges.data():
            coords_i = list(mol_graph.graph.nodes[i]["coords"])
            coords_j = list(mol_graph.graph.nodes[j]["coords"])
            if not is_ob_bonds(coords_i, coords_j):
                idxi = find_ob_index(coords_i)
                idxj = find_ob_index(coords_j)
                self.add_bond(idxi, idxj, order=0)


# TODO should let Molecule not dependent on db_entry. The needed property from db_entry
#  should be passed in at init
class Molecule:
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
        if self.enthalpy != None and self.entropy != None:
            return (
                self.energy * 27.21139
                + self.enthalpy * 0.0433641
                - T * self.entropy * 0.0000433641
            )
        else:
            return None

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
                for atom in ob.OBMolAtomDFSIter(self.ob_mol):
                    c = [atom.GetX(), atom.GetY(), atom.GetZ()]
                    if np.allclose(c, coords):
                        self._graph_idx_to_ob_idx_map[graph_idx] = atom.GetIdx()
                if graph_idx not in self.graph_idx_to_ob_idx_map:
                    raise Exception("atom not found.")
        return self._graph_idx_to_ob_idx_map

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
                    [edge], allow_reverse=False, alterations=None
                )
                sub_mols[edge] = new_mgs
            except MolGraphSplitError:  # cannot split, i.e. open ring
                new_mg = copy.deepcopy(self.mol_graph)
                idx1, idx2 = edge
                new_mg.break_edge(idx1, idx2, allow_reverse=True)
                sub_mols[edge] = [new_mg]
        return sub_mols

    def write(self, filename=None, file_format="sdf"):
        if filename is not None:
            create_directory(filename)
        self.ob_mol.SetTitle(str(self.id))
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

    def to_molecules(self, optimized=True, purify=True):
        """
        Convert data entries to molecules.

        Args:
            optimized (bool): If True, build from optimized molecule. otherwise build
                from initial molecule.
            purify (bool): If True, return unique ones, i.e. remove ones that have higher
                free energy. Uniqueness is determined isomorphism, charge, and spin
                multiplicity. If `optimized` is False, this will be set to `False`
                internally.

        Returns:
            A list of Molecule object
        """

        logger.info("Start converting DB entries to molecules...")

        n_unsuccessful_entry = 0
        n_unconnected_mol = 0
        n_not_unique_mol = 0
        mols = []
        if not optimized:
            purify = False
        for entry in self.entries:
            try:
                m = Molecule(entry, optimized=optimized)
                if purify:
                    idx = -1
                    # check for connectivity
                    if not nx.is_weakly_connected(m.graph):
                        n_unconnected_mol += 1
                        continue
                    # check for isomorphism
                    for i_p_m, p_m in enumerate(mols):
                        # TODO check whether spin_multiplicity is needed
                        if (
                            m.mol_graph.isomorphic_to(p_m.mol_graph)
                            and m.charge == p_m.charge
                            and m.spin_multiplicity == p_m.spin_multiplicity
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

        return mols

    @staticmethod
    def create_sdf_csv_dataset(
        molecules, structure_name="electrolyte.sdf", label_name="electrolyte.csv"
    ):
        logger.info(
            "Start writing dataset to files: {} and {}".format(structure_name, label_name)
        )

        with open(structure_name, "w") as fx, open(label_name, "w") as fy:

            fy.write("mol,property_1\n")

            for i, m in enumerate(molecules):
                entry = m.db_entry

                # conn = m.get_connectivity()
                # species = m.get_species()
                # coords = m.get_coords()
                # bond_order = m.get_bond_order()
                # print('conn', conn)
                # print('speices', species)
                # print('coords', coords)
                # print('bond_order', bond_order)

                sdf = m.write(file_format="sdf")
                smiles = m.write(file_format="smi")
                print("id", m.id, "smiles", smiles, "formula", m.formula, "\nsdf", sdf)

                fname = "images/{}_{}.svg".format(m.formula, i)
                m.draw(fname, show_atom_idx=True)

                fx.write(sdf)
                fy.write("{},{}\n".format(m.id, m.entropy))


class Reaction:
    """
    A reaction that only has one bond break or the type ``A -> B + C``
    (break a bond not in a ring) or ``A -> D`` (break a bond in a ring)

    Args:
        reactants: a list of Molecules
        products: a list of Molecules
        broken_bond (tuple): index indicating the broken bond in the mol_graph
    """

    # NOTE most methods in this class only works for A->B and A->B+C type reactions

    def __init__(self, reactants, products, broken_bond=None):
        assert len(reactants) == 1, "incorrect number of reactants, should be 1"
        assert 1 <= len(products) <= 2, "incorrect number of products, should be 1 or 2"
        self.reactants = reactants
        self.products = self._order_molecules(products)
        self.broken_bond = broken_bond

    def get_broken_bond(self):
        if self.broken_bond is None:
            if len(self.products) == 1:
                bond = is_valid_A_to_B_reaction(self.reactants[0], self.products[0])
            elif len(self.products) == 2:
                bond = is_valid_A_to_B_C_reaction(self.reactants[0], self.products)
            if bond is None:
                raise RuntimeError(
                    "invalid reaction (cannot break a reactant bond to get products)"
                )
            self.broken_bond = bond

        return self.broken_bond

    def get_broken_bond_attr(self):
        """
        Returns a dict of the species and bond order of the broken bond.
        """
        graph = self.reactants[0].graph
        u, v = self.get_broken_bond()
        spec_u = graph.nodes[u]["specie"]
        spec_v = graph.nodes[v]["specie"]
        # TODO, we temporarily need the try except block because the
        #  metal_edge_extender does not add weight/get
        try:
            order = graph.get_edge_data(u, v, key=0)[
                "weight"
            ]  # key=0 because of MultiGraph
        except KeyError:
            order = 0
        return {"species": sorted([spec_u, spec_v]), "order": order}

    def get_reaction_free_energy(self):
        energy = 0
        for mol in self.reactants:
            energy -= mol.free_energy
        for mol in self.products:
            energy += mol.free_energy
        return energy

    @staticmethod
    def _order_molecules(mols):
        """
        Order the molecules according to the below rules (in order):
        1. molecular mass
        2. number of atoms
        3. number of bonds
        4. alphabetical formula
        5. diameter of molecule graph, i.e. largest distance for node to node
        6. charge

        Args:
            mols: a list of molecules.

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
            pa = len(a.atoms)
            pb = len(b.atoms)
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

        if len(mols) == 1:
            return mols

        out = order_by_weight(*mols)
        if out is not None:
            return out
        out = order_by_natoms(*mols)
        if out is not None:
            return out
        out = order_by_nbonds(*mols)
        if out is not None:
            return out
        out = order_by_formula(*mols)
        if out is not None:
            return out
        out = order_by_diameter(*mols)
        if out is not None:
            return out
        a, b = mols
        if a.mol_graph.isomorphic_to(b.mol_graph):
            out = order_by_charge(*mols)  # e.g. H+ and H-
            if out is not None:
                return out
            else:
                return mols  # two exactly the same molecules
        raise RuntimeError("Cannot order molecules")

    def as_dict(self):
        d = {
            "reactants": ["{}_{}".format(m.formula, m.id) for m in self.reactants],
            "products": ["{}_{}".format(m.formula, m.id) for m in self.products],
            "charge": [m.charge for m in self.reactants + self.products],
            "broken_bond": self.broken_bond,
            "bond_energy": self.get_reaction_free_energy(),
        }
        return d

    def to_file(self, filename):
        mols = self.reactants + self.products
        for m in mols:
            m.make_picklable()
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "reactants": self.reactants,
            "products": self.products,
            "broken_bond": self.broken_bond,
        }
        pickle_dump(d, filename)

    @classmethod
    def from_file(cls, filename):
        d = pickle_dump(filename)
        return cls(d["reactants"], d["products"], d["broken_bond"])


class ReactionExtractor:
    def __init__(self, molecules, reactions=None):
        self.molecules = molecules or self._get_molecules_from_reactions(reactions)
        self.reactions = reactions

        self.buckets = None
        self.bucket_keys = None

    def get_molecule_properties(self, keys):
        values = defaultdict(list)
        for m in self.molecules:
            for k in keys:
                values[k].append(getattr(m, k))
        return values

    def bucket_molecules(self, keys=["formula", "charge"]):
        """
        Classify molecules into nested dictionaries according to molecule properties
        specified in ``keys``.

        Args:
            keys (list of str): each str should be a molecule property.

        Returns:
            nested dictionary of molecules classified according to keys.
        """
        logger.info("Start bucketing molecules...")

        num_keys = len(keys)
        buckets = {}
        for m in self.molecules:
            b = buckets
            for i, k in enumerate(keys):
                v = getattr(m, k)
                if i == num_keys - 1:
                    b.setdefault(v, []).append(m)
                else:
                    b.setdefault(v, {})
                b = b[v]

        self.bucket_keys = keys
        self.buckets = buckets

    def extract_A_to_B_style_reaction(self):
        """
        Return a list of A -> B reactions.
        """
        logger.info("Start extracting A -> B style reactions")

        if self.buckets is None:
            self.bucket_molecules(keys=["formula", "charge"])

        A2B = []
        i = 0
        for formula, entries_formula in self.buckets.items():
            i += 1
            if i % 10000 == 0:
                print("@@flag running bucket", i)
            for charge, entries_charges in entries_formula.items():
                for A, B in itertools.permutations(entries_charges, 2):
                    bond = is_valid_A_to_B_reaction(A, B)
                    if bond is not None:
                        A2B.append(Reaction([A], [B], bond))
        self.reactions = A2B

        return A2B

    def extract_A_to_B_C_style_reaction(self):
        """
        Return a list of A -> B + C reactions.
        """
        logger.info("Start extracting A -> B + C style reactions")

        if self.buckets is None:
            self.bucket_molecules(keys=["formula", "charge"])

        fcmap = self._get_formula_composition_map(self.molecules)
        A2BC = []
        i = 0
        for (
            (formula_A, entries_formula_A),
            (formula_B, entries_formula_B),
            (formula_C, entries_formula_C),
        ) in itertools.product(self.buckets.items(), repeat=3):

            i += 1
            if i % 10000 == 0:
                print("@@flag running bucket", i)

            composition_A = fcmap[formula_A]
            composition_B = fcmap[formula_B]
            composition_C = fcmap[formula_C]

            if not self._is_valid_A_to_B_C_composition(
                composition_A, composition_B, composition_C
            ):
                continue

            reaction_ids = []
            for (
                (charge_A, entries_charge_A),
                (charge_B, entries_charge_B),
                (charge_C, entries_charge_C),
            ) in itertools.product(
                entries_formula_A.items(),
                entries_formula_B.items(),
                entries_formula_C.items(),
            ):
                if not self._is_valid_A_to_B_C_charge(charge_A, charge_B, charge_C):
                    continue

                for A, B, C in itertools.product(
                    entries_charge_A, entries_charge_B, entries_charge_C
                ):
                    bond = is_valid_A_to_B_C_reaction(A, [B, C])
                    if bond is not None:
                        ids = set([A.id, B.id, C.id])
                        # remove repeating reactions (e.g. A->B+C and A->C+B)
                        if ids not in reaction_ids:
                            A2BC.append(Reaction([A], [B, C], bond))
                            reaction_ids.append(ids)

        self.reactions = A2BC

        return A2BC

    def extract_one_bond_break(self):
        """
        Extract all reactions that only has one bond break or the type ``A -> B + C``
        (break a bond not in a ring) or ``A -> D`` (break a bond in a ring)

        Returns:
            A list of reactions.
        """
        A2B = self.extract_A_to_B_style_reaction()
        A2BC = self.extract_A_to_B_C_style_reaction()
        self.reactions = A2B + A2BC

        return A2B, A2BC

    def group_by_reactant_and_charge(
        self, string_reactant_index=False, babel_bond_indices=True
    ):
        """
        Group all the reactions according to the reactant and the charge of the
        reactant and products.

        Args:
            string_reactant_index: If True, the reactant id (a string) is used as the
                key for the outer dict, otherwise, the reactant instance is used.
            babel_bond_indices: If True, babel atoms (indices) will be used as the key
                to denote broken bond for the inner dict, otherwise, mol_graph nodes
                will be used.

        Returns:
            A dict of dict of dict. The outer dict has reactant index as the key,
            the middle dict has charges (a tuple) as the key and the inner dict has bond
            indices (a tuple) as the key and bond attributes (a dict of energy,
            bond order, ect.).


        """

        def get_bond_idx(bond, reactant, babel_bond_indices):
            if babel_bond_indices:
                idx0 = reactant.graph_idx_to_ob_idx_map[bond[0]]
                idx1 = reactant.graph_idx_to_ob_idx_map[bond[1]]
                bond_indices = tuple(sorted([idx0, idx1]))
            else:
                bond_indices = bond
            return bond_indices

        grouped_reactions = defaultdict(list)
        for rxn in self.reactions:
            grouped_reactions[rxn.reactants[0]].append(rxn)

        groups = dict()
        for reactant, reactions in grouped_reactions.items():
            if string_reactant_index:
                reactant_idx = reactant.formula + "_" + reactant.id
            else:
                reactant_idx = reactant
            groups[reactant_idx] = dict()
            for rxn in reactions:
                charge = tuple([m.charge for m in (rxn.reactants + rxn.products)])
                if charge not in groups[reactant_idx]:
                    groups[reactant_idx][charge] = OrderedDict()
                    for bond in reactant.graph.edges():
                        bond_indices = get_bond_idx(bond, reactant, babel_bond_indices)
                        groups[reactant_idx][charge][bond_indices] = None
            for rxn in reactions:
                charge = tuple([m.charge for m in (rxn.reactants + rxn.products)])
                bond_indices = get_bond_idx(
                    rxn.get_broken_bond(), reactant, babel_bond_indices
                )
                groups[reactant_idx][charge][bond_indices] = rxn.as_dict()

        return groups

    def bond_energies_to_file(
        self, filename, string_reactant_index=True, babel_bond_indices=True
    ):
        groups = self.group_by_reactant_and_charge(
            string_reactant_index, babel_bond_indices
        )
        for m in self.molecules:
            m.make_picklable()
        yaml_dump(groups, filename)

    def create_struct_label_dataset(
        self, struct_name="sturct.sdf", label_name="label.txt"
    ):
        def get_rdf_bond_order(sdf):
            lines = sdf.split("\n")
            split_3 = lines[3].split()
            natoms = int(split_3[0])
            nbonds = int(split_3[1])
            bonds = []
            for line in lines[4 + natoms : 4 + natoms + nbonds]:
                bonds.append(tuple(sorted([int(i) for i in line.split()[:2]])))
            return bonds

        logger.info(
            "Start writing reactions to files: {} and {}".format(struct_name, label_name)
        )

        reactants_bond_energies = self.group_by_reactant_and_charge(
            string_reactant_index=False, babel_bond_indices=True
        )

        struct_name, label_name = expand_path(struct_name), expand_path(label_name)
        with open(struct_name, "w") as f_struct, open(label_name, "w") as f_label:
            f_label.write(
                "# Each line lists the bond energies of a molecule. "
                "It contains four parts: "
                "1) the number of substances N (i.e. number of reactants and products) "
                "2) N numbers of the charge of the substances "
                "3) the remaining part is equal to 2 times the number of bonds, "
                "where the first half values are bond energies and "
                "4) the next half values are indicators "
                " (0 or 1) to specify whether the bond energy exist in the dataset. a "
                "value of 0 means the corresponding bond energy will be ignored, "
                "no matter what its value is.\n"
            )
            for reactant, reactions in reactants_bond_energies.items():
                # struct
                sdf = reactant.write(file_format="sdf")
                f_struct.write(sdf)

                sdf_bonds = get_rdf_bond_order(sdf)

                # label
                for charge, rxns in reactions.items():
                    f_label.write("{} ".format(len(charge)))
                    for c in charge:
                        f_label.write("{} ".format(c))
                    f_label.write("   ")

                    # write bond energies in the same order as sdf file
                    bond_energies = dict()
                    for bond, attr in rxns.items():
                        if attr is not None:
                            bond_energies[bond] = attr["bond_energy"]
                        else:
                            bond_energies[bond] = None
                    for bond in sdf_bonds:
                        energy = bond_energies[bond]
                        if energy is None:
                            f_label.write("0.0 ")
                        else:
                            f_label.write("{:.15g} ".format(energy))
                    f_label.write("   ")

                    # write bond energy indicator
                    for bond in sdf_bonds:
                        energy = bond_energies[bond]
                        if energy is None:
                            f_label.write("0 ")
                        else:
                            f_label.write("1 ")
                    f_label.write("\n")

    @staticmethod
    def _get_formula_composition_map(mols):
        fcmap = dict()
        for m in mols:
            fcmap[m.formula] = m.composition_dict
        return fcmap

    @staticmethod
    def _is_even_composition(composition):
        for spec, amt in composition.items():
            if int(amt) % 2 != 0:
                return False
        return True

    @staticmethod
    def _is_valid_A_to_B_C_composition(composition1, composition2, composition3):
        combined23 = defaultdict(float)
        for k, v in composition2.items():
            combined23[k] += v
        for k, v in composition3.items():
            combined23[k] += v

        sorted1 = sorted(composition1.keys())
        sorted23 = sorted(combined23.keys())
        if sorted1 != sorted23:
            return False
        for k in sorted1:
            if int(composition1[k]) != int(combined23[k]):
                return False
        return True

    @staticmethod
    def _get_molecules_from_reactions(reactions):
        mols = []
        for r in reactions:
            mols.extend(r.reactants + r.products)
        return list(set(mols))

    @staticmethod
    def _is_valid_A_to_B_C_charge(charge1, charge2, charge3):
        return charge1 == charge2 + charge3

    def to_file(self, filename="rxns.pkl"):
        logger.info("Start writing reactions to file: {}".format(filename))

        for m in self.molecules:
            m.make_picklable()
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "molecules": self.molecules,
            "reactions": self.reactions,
        }
        pickle_dump(d, filename)

    @classmethod
    def from_file(cls, filename):
        d = pickle_load(filename)
        logger.info(
            "{} reactions loaded from file: {}".format(len(d["reactions"]), filename)
        )
        return cls(d["molecules"], d["reactions"])

    def to_file_by_ids(self, filename="rxns.json"):
        logger.info("Start writing reactions by ids to file: {}".format(filename))
        reaction_ids = []
        for i, r in enumerate(self.reactions):
            reaction_ids.append(r.as_dict())
        with open(filename, "w") as f:
            json.dump(reaction_ids, f)

    @classmethod
    def from_file_by_ids(cls, filename, db_path):
        logger.info("Start loading reactions by ids from file: {}".format(filename))

        db = DatabaseOperation.from_file(db_path)
        mols = db.to_molecules(purify=True)
        id_to_mol_map = {m.id: m for m in mols}

        with open(filename, "r") as f:
            reactions = json.load(f)

        rxns = []
        for r in tqdm(reactions):
            reactants = [id_to_mol_map[i] for i in r["reactants"]]
            products = [id_to_mol_map[i] for i in r["products"]]
            broken_bond = r["broken_bond"]
            rxn = Reaction(reactants, products, broken_bond)
            rxns.append(rxn)

        logger.info("Finish loading {} reactions".format(len(reactions)))

        return cls(mols, rxns)


def is_valid_A_to_B_reaction(reactant, product):
    """
    A -> B
    Args:
        reactant: mol
        product: mol

    Returns:
        A tuple of the bond indices, if this is valid reaction;
        None, otherwise.
    """
    for edge, mgs in reactant.fragments.items():
        if len(mgs) == 1 and mgs[0].isomorphic_to(product.mol_graph):
            return edge
    return None


def is_valid_A_to_B_C_reaction(reactant, products):
    """
    A -> B + C
    Args:
        reactant: mol
        products: list of mols

    Returns:
        A tuple of the bond indices, if this is valid reaction;
        None, otherwise.
    """
    for edge, mgs in reactant.fragments.items():
        if len(mgs) == 2:
            if (
                mgs[0].isomorphic_to(products[0].mol_graph)
                and mgs[1].isomorphic_to(products[1].mol_graph)
            ) or (
                mgs[0].isomorphic_to(products[1].mol_graph)
                and mgs[1].isomorphic_to(products[0].mol_graph)
            ):
                return edge
    return None


class UnsuccessfulEntryError(Exception):
    def __init__(self):
        pass
