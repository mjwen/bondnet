import pickle
import copy
import itertools
import json
import logging
from tqdm import tqdm
from collections import defaultdict
from atomate.qchem.database import QChemCalcDb
import pymatgen
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.analysis.fragmenter import metal_edge_extender
from pymatgen.io.babel import BabelMolAdaptor
import openbabel as ob
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from gnn.data.utils import create_directory

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DatabaseOperation:
    def __init__(self, entries):
        self.entries = entries

    @classmethod
    def from_query(cls, db_file="/Users/mjwen/Applications/mongo_db_access/sam_db.json"):
        """
        Query a (Sam's) database to pull all the molecules.
        """
        logger.info('Start building database from query...')

        # Create a json file that contains the credentials
        mmdb = QChemCalcDb.from_db_file(db_file, admin=True)

        # This contains all the production jobs(wb97xv/def2-tzvppd/smd(LiEC parameters)).
        # Every target_entries[i] is a dictionary of all the information for one job.
        entries = list(mmdb.collection.find({"tags.class": 'smd_production'}))
        return cls(entries)

    @classmethod
    def from_file(cls, filename='database.pkl'):
        """
        Recover a dumped database entries.

        Returns:
            A list of database entries.
        """
        logger.info('Start building database from file: {}'.format(filename))

        with open(filename, 'rb') as f:
            entries = pickle.load(f)
        return cls(entries)

    @staticmethod
    def filter(entries, keys, value):
        """
        Filter out all database entries whose value match the give value.

        Args:
            entries: database entries.
            keys: list of dictionary keys
            value: the value to which to match

        Returns:
            list of database entries that are filtered out
        """
        results = []
        for entry in entries:
            v = entry
            for k in keys:
                v = v[k]
            if v == value:
                results.append(entry)
        return results

    @staticmethod
    def to_molecules(entries):
        mols = []
        for entry in entries:
            try:
                mols.append(Molecule(entry))
            except KeyError:
                pass
        return mols

    @staticmethod
    def to_file(entries, size=None, filename='database.pkl'):
        """
        Dump a list of database molecule entries to disk.
        This is purely for efficiency stuff, since the next time we want to query the
        database, we can simply load the dumped one instead of query the actual database.
        """
        logger.info('Start writing database to file: {}'.format(filename))

        size = size or -1
        entries = entries[:size]
        with open(filename, 'wb') as f:
            pickle.dump(entries, f)

    @staticmethod
    def create_sdf_csv_dataset(
        entries, structure_name='electrolyte.sdf', label_name='electrolyte.csv'
    ):

        logger.info(
            'Start writing dataset to files: {} and {}'.format(structure_name, label_name)
        )

        with open(structure_name, 'w') as fx, open(label_name, 'w') as fy:

            fy.write('mol,property_1\n')

            for i, entry in enumerate(entries):
                try:
                    m = Molecule(entry)
                except KeyError:
                    continue

                # conn = m.get_connectivity()
                # species = m.get_species()
                # coords = m.get_coords()
                # bond_order = m.get_bond_order()
                # print('conn', conn)
                # print('speices', species)
                # print('coords', coords)
                # print('bond_order', bond_order)

                sdf = m.write(file_format='sdf')
                smiles = m.write(file_format='smi')
                print('id', m.id, 'smiles', smiles, 'formula', m.formula, '\nsdf', sdf)

                fname = 'images/{}_{}.svg'.format(m.formula, i)
                m.draw(fname, show_atom_idx=True)

                fx.write(sdf)
                fy.write('{},{}\n'.format(m.id, m.entropy))


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
                raise Exception('bond exists')

        self._obmol.AddBond(idx1, idx2, order)

    @classmethod
    def from_molecule_graph(cls, mol_graph):
        if not isinstance(mol_graph, MoleculeGraph):
            raise ValueError('not get mol graph')
        self = cls(mol_graph.molecule)
        self._add_missing_bond(mol_graph)

        return self

    def _add_missing_bond(self, mol_graph):
        def is_ob_bonds(coords1, coords2):

            for i, atom in enumerate(ob.OBMolAtomDFSIter(self._obmol)):
                cdi = [atom.GetX(), atom.GetY(), atom.GetZ()]
                if cdi != coords1:
                    continue
                for neighbor in ob.OBAtomAtomIter(atom):
                    cdn = [neighbor.GetX(), neighbor.GetY(), neighbor.GetZ()]
                    if cdn == coords2:
                        return True
            return False

        def find_ob_index(coords):
            for atom in ob.OBMolAtomDFSIter(self._obmol):
                c = [atom.GetX(), atom.GetY(), atom.GetZ()]
                if c == coords:
                    return atom.GetIdx()

        for i, j, attr in mol_graph.graph.edges.data():
            coords_i = list(mol_graph.graph.nodes[i]['coords'])
            coords_j = list(mol_graph.graph.nodes[j]['coords'])
            if not is_ob_bonds(coords_i, coords_j):
                idxi = find_ob_index(coords_i)
                idxj = find_ob_index(coords_j)
                self.add_bond(idxi, idxj, order=0)


# TODO should let Molecule not dependent on db_entry. The needed property from db_entry
#  should be passed in at init
class Molecule:
    def __init__(self, db_entry, use_metal_edge_extender=True):
        self.db_entry = db_entry
        try:
            self.mol = pymatgen.Molecule.from_dict(
                db_entry['output']['optimized_molecule']
            )
        except KeyError:
            print('error seen for id:', db_entry['_id'])
            raise KeyError

        self.mol_graph = MoleculeGraph.with_local_env_strategy(
            self.mol, OpenBabelNN(order=True), reorder=False, extend_structure=False
        )
        if use_metal_edge_extender:
            self.mol_graph = metal_edge_extender(self.mol_graph)
        self.ob_adaptor = BabelMolAdaptor2.from_molecule_graph(self.mol_graph)

    @property
    def pymatgen_mol(self):
        return self.mol

    @property
    def ob_mol(self):
        return self.ob_adaptor.openbabel_mol

    @property
    def pybel_mol(self):
        return self.ob_adaptor.pybel_mol

    @property
    def rdkit_mol(self):
        sdf = self.write(file_format='sdf')
        return Chem.MolFromMolBlock(sdf)

    @property
    def id(self):
        return str(self._get_property_from_db_entry(['_id']))

    @property
    def energy(self):
        return self._get_property_from_db_entry(['output', 'final_energy'])

    @property
    def entropy(self):
        return self._get_property_from_db_entry(['output', 'entropy'])

    @property
    def enthalpy(self):
        return self._get_property_from_db_entry(['output', 'enthalpy'])

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
        return f.replace(' ', '')

    @property
    def composition_dict(self):
        return self.pymatgen_mol.composition.as_dict()

    def get_connectivity(self):
        conn = []
        for u, v in self.graph.edges():
            conn.append([u, v])
        return conn

    def get_species(self):
        return self._get_node_attr('specie')

    def get_coords(self):
        return self._get_node_attr('coords')

    def get_bond_order(self):
        return self._get_edge_attr('weight')

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

    def write(self, filename=None, file_format='sdf'):
        self.ob_mol.SetTitle(str(self.id))
        return self.pybel_mol.write(file_format, filename, overwrite=True)

    def draw(self, filename=None, draw_2D=True, show_atom_idx=False):
        sdf = self.write(file_format='sdf')
        m = Chem.MolFromMolBlock(sdf)
        if draw_2D:
            AllChem.Compute2DCoords(m)
        if show_atom_idx:
            atoms = [m.GetAtomWithIdx(i) for i in range(m.GetNumAtoms())]
            _ = [a.SetAtomMapNum(a.GetIdx() + 1) for a in atoms]
        filename = filename or 'mol.svg'
        filename = create_directory(filename)
        Draw.MolToFile(m, filename)


class Reaction:
    """
    A reaction that only has one bond break or the type ``A -> B + C``
    (break a bond not in a ring) or ``A -> D`` (break a bond in a ring)

    Args:
        reactants:
    """

    def __init__(self, reactants, products, broken_bond=None):
        assert len(reactants) == 1, 'incorrect number of reactants, should be 1'
        assert 1 <= len(products) <= 2, 'incorrect number of products, should be 1 or 2'
        self.reactants = reactants
        self.products = products
        self.broken_bond = broken_bond or self.get_broken_bond()
        self._tags = {}

    def as_dict(self):
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "reactants": [m.id for m in self.reactants],
            "products": [m.id for m in self.products],
            "broken_bond": self.broken_bond,
        }
        return d

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, keys, values):
        self._tags = dict(zip(keys, values))

    @tags.setter
    def tags(self, key_val_dict):
        self._tags = dict(key_val_dict)

    def get_broken_bond(self):
        if self.broken_bond is None:
            if len(self.products) == 1:
                bond = is_valid_A_to_B_reaction(self.reactants[0], self.products[0])
            elif len(self.products) == 2:
                bond = is_valid_A_to_B_C_reaction(self.reactants[0], self.products)
            if bond is None:
                raise RuntimeError(
                    'invalid reaction (cannot break a reactant bond to get products)'
                )
            self.broken_bond = bond

    def get_broken_bond_attr(self):
        """
        Returns a dict of the species and bond order of the broken bond.
        """
        graph = self.reactants[0].graph
        u, v = self.broken_bond
        spec_u = graph.nodes[u]['specie']
        spec_v = graph.nodes[v]['specie']
        # TODO, we temporarily need the try except block because the
        #  metal_edge_extender does not add weight/get
        try:
            order = graph.get_edge_data(u, v, key=0)[
                'weight'
            ]  # key=0 because of MultiGraph
        except KeyError:
            order = 0
        return {'species': sorted([spec_u, spec_v]), 'order': order}

    def get_reaction_free_energy(self):
        energy = 0
        for mol in self.reactants:
            energy -= mol.free_energy
        for mol in self.products:
            energy += mol.free_energy
        return energy


def fragment_molecule(mol_graph, mol_id=None):
    """
    Fragment molecule by breaking ONE bond.

    Args:
        mol_graph:

    Returns:
        A dictionary with key of bond index (a tuple (idx1, idx2)), and value of the new
        mol_graphs. Could be empty if the mol has no bonds.
    """
    mg = mol_graph
    sub_mols = {}
    for edge in mg.graph.edges():
        try:
            new_mgs = mg.split_molecule_subgraphs(
                [edge], allow_reverse=False, alterations=None
            )
            sub_mols[edge] = new_mgs
        except MolGraphSplitError:
            # print('cannot split:', mol_id)
            new_mg = copy.deepcopy(mg)
            idx1, idx2 = edge
            new_mg.break_edge(idx1, idx2, allow_reverse=True)
            sub_mols[edge] = [new_mg]
    return sub_mols


def is_valid_A_to_B_reaction(reactant, product, reactant_fragments=None):
    """
    A -> B
    Args:
        reactant: mol
        product: mol

    Returns:
        A tuple of the bond indices, if this is valid reaction;
        None, otherwise.
    """
    fragments = reactant_fragments or fragment_molecule(reactant.mol_graph, reactant.id)
    for edge, mgs in fragments.items():
        if len(mgs) == 1 and mgs[0].isomorphic_to(product.mol_graph):
            return edge
    return None


def is_valid_A_to_B_C_reaction(reactant, products, reactant_fragments=None):
    """
    A -> B + C
    Args:
        reactant: mol
        products: list of mols

    Returns:
        A tuple of the bond indices, if this is valid reaction;
        None, otherwise.
    """
    fragments = reactant_fragments or fragment_molecule(reactant.mol_graph, reactant.id)
    for edge, mgs in fragments.items():
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


class ReactionExtractor:
    def __init__(self, molecules):
        self.mols = molecules

    def get_molecule_properties(self, keys):
        values = defaultdict(list)
        for m in self.mols:
            for k in keys:
                values[k].append(getattr(m, k))
        return values

    def bucket_molecules(
        self, keys=['formula', 'charge', 'spin_multiplicity'], purify=True
    ):
        """
        Classify molecules into nested dictionaries according to molecule properties
        specified in ``keys``.

        Args:
            keys (list of str): each str should be a molecule property.
            purify (bool): Purify each bucket such that we only have one instance
                of each molecule with the lowest energy. (i.e. remove molecules with
                the same connectivity but different atom positions.

        Returns:
            nested dictionary of molecules classified according to keys.
        """
        logger.info('Start bucketing molecules...')

        num_keys = len(keys)
        buckets = {}
        for m in self.mols:
            b = buckets
            for i, k in enumerate(keys):
                v = getattr(m, k)
                if i == num_keys - 1:
                    b.setdefault(v, []).append(m)
                else:
                    b.setdefault(v, {})
                b = b[v]

        if purify:
            buckets = self.purify_buckets(buckets)

        self.bucket_keys = keys
        self.buckets = buckets

        return buckets

    def purify_buckets(self, buckets):
        """
        Purify each bucket such that we only have one instance of each molecule with
        the lowest energy. (i.e. remove molecules with the same connectivity but
        different atom positions.
        Returns:
            A list of molecules.
        """

        def iter_nested_dict(d, new_d):
            for k, v in d.items():
                if isinstance(v, dict):
                    new_d[k] = {}
                    iter_nested_dict(v, new_d[k])
                else:
                    new_d[k] = self.purify(v)

        new_buckets = {}
        iter_nested_dict(buckets, new_buckets)

        return new_buckets

    @staticmethod
    def purify(molecules):
        """
        Given a list of molecules, return unique ones, i.e. remove ones that are
        isomorphic and have higher free energy.
        """
        purified = []
        for mol in molecules:
            idx = -1
            for i_p_mol, p_mol in enumerate(purified):
                if mol.mol_graph.isomorphic_to(p_mol.mol_graph):
                    idx = i_p_mol
                    break
            if idx >= 0:
                if mol.free_energy < purified[idx].free_energy:
                    purified[idx] = mol
            else:
                purified.append(mol)
        return purified

    def extract_A_to_B_style_reaction(self, mol_fragments=None):
        logger.info('Start extracting A -> B style reactions')

        mol_fragments = mol_fragments or self.fragment_molecules(self.mols)

        A_to_B_rxns = []
        i = 0
        for formula, entries_formula in self.buckets.items():
            i += 1
            if i % 100 == 0:
                print('@@flag1 running bucket', i)
            for charge, entries_charges in entries_formula.items():
                for A, B in itertools.permutations(entries_charges, 2):
                    bond = is_valid_A_to_B_reaction(A, B, mol_fragments[A.id])
                    if bond is not None:
                        A_to_B_rxns.append(Reaction([A], [B], bond))

        return A_to_B_rxns

    def extract_A_to_B_C_style_reaction(self, mol_fragments=None, fcmap=None):
        logger.info('Start extracting A -> B + C style reactions')

        mol_fragments = mol_fragments or self.fragment_molecules(self.mols)
        fcmap = fcmap or self.get_formula_composition_map(self.mols)

        i = 0
        A_to_B_C_rxns = []
        for (
            (formula_A, entries_formula_A),
            (formula_B, entries_formula_B),
            (formula_C, entries_formula_C),
        ) in itertools.product(self.buckets.items(), repeat=3):

            i += 1
            if i % 100 == 0:
                print('@@flag1 running bucket', i)

            composition_A = fcmap[formula_A]
            composition_B = fcmap[formula_B]
            composition_C = fcmap[formula_C]

            if not self._is_valid_A_to_B_C_composition(
                composition_A, composition_B, composition_C
            ):
                continue

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
                    bond = is_valid_A_to_B_C_reaction(A, [B, C], mol_fragments[A.id])
                    if bond is not None:
                        A_to_B_C_rxns.append(Reaction([A], [B, C], bond))

        return A_to_B_C_rxns

    def extract_one_bond_break(self):
        """
        Extract all reactions that only has one bond break or the type ``A -> B + C``
        (break a bond not in a ring) or ``A -> D`` (break a bond in a ring)

        Returns:

        """
        mol_fragments = self.fragment_molecules(self.mols)
        fcmap = self.get_formula_composition_map(self.mols)

        A2B = self.extract_A_to_B_style_reaction(mol_fragments, fcmap)
        A2BC = self.extract_A_to_B_C_style_reaction(mol_fragments, fcmap)

        return A2B, A2BC

    @staticmethod
    def to_file(reactions, filename='rxns.json'):
        logger.info('Start writing reactions to file: {}'.format(filename))

        reaction_ids = []
        for i, r in enumerate(reactions):
            if i % 100 == 0:
                print('@@flag, as_dict:', i)
            reaction_ids.append(r.as_dict())

        with open(filename, 'w') as f:
            json.dump(reaction_ids, f)

    @staticmethod
    def fragment_molecules(mols):
        fragments = dict()
        for m in mols:
            fragments[m.id] = fragment_molecule(m.mol_graph, m.id)
        return fragments

    @staticmethod
    def get_formula_composition_map(mols):
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
    def _is_valid_A_to_B_C_charge(charge1, charge2, charge3):
        return charge1 == charge2 + charge3


def load_extracted_reactions(filename, db_path):
    entries = DatabaseOperation.from_file(db_path).entries
    mols = DatabaseOperation.to_molecules(entries)
    id_to_mol_map = {m.id: m for m in mols}

    logger.info('Start loading extracted reactions from file: {}'.format(filename))
    with open(filename, 'r') as f:
        reactions = json.load(f)
    logger.info(
        'Finish loading {} extracted reactions from file: {}'.format(
            len(reactions), filename
        )
    )

    logger.info('Start recover reactions from ids...')
    rxns = []
    for r in tqdm(reactions):
        reactants = [id_to_mol_map[i] for i in r['reactants']]
        products = [id_to_mol_map[i] for i in r['products']]
        broken_bond = r['broken_bond']
        rxn = Reaction(reactants, products, broken_bond)
        rxns.append(rxn)

    return rxns
