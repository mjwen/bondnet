import copy
import itertools
import logging
import warnings
import json
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
from gnn.data.utils import create_directory, pickle_dump, pickle_load

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
        logger.info('Start loading database from file: {}'.format(filename))
        entries = pickle_load(filename)
        return cls(entries)

    def to_file(self, size=None, filename='database.pkl'):
        """
        Dump a list of database molecule entries to disk.
        This is purely for efficiency stuff, since the next time we want to query the
        database, we can simply load the dumped one instead of query the actual database.
        """
        logger.info('Start writing database to file: {}'.format(filename))
        size = size or -1
        entries = self.entries[:size]
        pickle_load(entries, filename)

    def filter(self, keys, value):
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
        for entry in self.entries:
            v = entry
            for k in keys:
                v = v[k]
            if v == value:
                results.append(entry)

        self.entries = results

    def to_molecules(self, purify=True):
        """
        Convert data entries to molecules.

        Args:
            purify (bool): If True, return unique ones, i.e. remove ones that have higher
                free energy. Uniqueness is determined isomorphism, charge, and spin
                multiplicity.

        Returns:
            A list of Molecule object
        """

        logger.info('Start converting DB entries to molecules...')

        n_not_opt_mol = 0
        n_not_unique_mol = 0
        mols = []
        for entry in self.entries:
            try:
                m = Molecule(entry)
                if purify:
                    idx = -1
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
            except KeyError:
                n_not_opt_mol += 1
                pass
        logger.info(
            'DB entry size: {}; Entries without `["output"]["optimized_molecule"]`'
            ': {}; Isomorphic entries: {}; Entries converted to molecules: {}.'.format(
                len(self.entries), n_not_opt_mol, n_not_unique_mol, len(mols)
            )
        )

        return mols

    @staticmethod
    def create_sdf_csv_dataset(
        molecules, structure_name='electrolyte.sdf', label_name='electrolyte.csv'
    ):
        logger.info(
            'Start writing dataset to files: {} and {}'.format(structure_name, label_name)
        )

        with open(structure_name, 'w') as fx, open(label_name, 'w') as fy:

            fy.write('mol,property_1\n')

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

        self.use_metal_edge_extender = use_metal_edge_extender

        self._mol_graph = None
        self._ob_adaptor = None
        self._fragments = None

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

    @property
    def fragments(self):
        if self._fragments is None:
            self._fragments = self._get_fragments()
        return self._fragments

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
            except MolGraphSplitError:
                # print('cannot split:', self.id)
                new_mg = copy.deepcopy(self.mol_graph)
                idx1, idx2 = edge
                new_mg.break_edge(idx1, idx2, allow_reverse=True)
                sub_mols[edge] = [new_mg]
        return sub_mols

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
        reactants: a list of Molecules
        products: a list of Molecules
        broken_bond (tuple): index indicating the broken bond in the mol_graph
    """

    def __init__(self, reactants, products, broken_bond=None):
        assert len(reactants) == 1, 'incorrect number of reactants, should be 1'
        assert 1 <= len(products) <= 2, 'incorrect number of products, should be 1 or 2'
        self.reactants = reactants
        self.products = products
        self.broken_bond = broken_bond

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

        return self.broken_bond

    def get_broken_bond_attr(self):
        """
        Returns a dict of the species and bond order of the broken bond.
        """
        graph = self.reactants[0].graph
        u, v = self.get_broken_bond()
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
        return cls(d['reactants'], d['products'], d['broken_bond'])


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

    def bucket_molecules(self, keys=['formula', 'charge']):
        """
        Classify molecules into nested dictionaries according to molecule properties
        specified in ``keys``.

        Args:
            keys (list of str): each str should be a molecule property.

        Returns:
            nested dictionary of molecules classified according to keys.
        """
        logger.info('Start bucketing molecules...')

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
        logger.info('Start extracting A -> B style reactions')

        if self.buckets is None:
            self.bucket_molecules(keys=['formula', 'charge'])

        A2B = []
        i = 0
        for formula, entries_formula in self.buckets.items():
            i += 1
            if i % 100 == 0:
                print('@@flag1 running bucket', i)
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
        logger.info('Start extracting A -> B + C style reactions')

        if self.buckets is None:
            self.bucket_molecules(keys=['formula', 'charge'])

        fcmap = self._get_formula_composition_map(self.molecules)
        A2BC = []
        i = 0
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
                    bond = is_valid_A_to_B_C_reaction(A, [B, C])
                    if bond is not None:
                        A2BC.append(Reaction([A], [B, C], bond))
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

    def group_by_reactant(self):
        """
        Group all the reactions according to the reactant.
        Many reactions will have the same reactant if we break different bond.
        Returns:
            A dict with reactant id as the key and a list of reactions (that have the
            same reactant) as the value.

        """
        grouped_reactions = defaultdict(list)
        for r in self.reactions:
            grouped_reactions[r.reactants[0].id].append(r)
        return grouped_reactions

    def get_reactants_bond_energies(self, ids=None):
        """
        Get the bond energy of reactants.

        Each bond energy is computed from the reaction energy, If there is no reaction
        associated with a bond, its bond energy is set to None.

        Args:
            ids: a list of string identifier specifying the reactants whose bond
                energies need to be returned. If ``None`` all are returned.

        Returns:
            A dict of dict. The outer dict has reactant instance as the key and the
            inner dict has bond indices (a tuple) as the key and bond energy (a float,
            could be None) as the value.
        """
        grouped_reactions = self.group_by_reactant()

        reactants_bond_energies = dict()
        for _, reactions in grouped_reactions.items():
            reactant = reactions[0].reactants[0]
            if ids is not None and reactant.id not in ids:
                continue
            bonds = reactant.graph.edges()
            energies = {bond: None for bond in bonds}
            for r in reactions:
                energies[r.get_broken_bond()] = r.get_reaction_free_energy()
            reactants_bond_energies[reactant] = energies

        if ids is not None and len(ids) != len(reactants_bond_energies):
            warnings.warn('bond energies for some molecules not available.')

        return reactants_bond_energies

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

    def to_file(self, filename='rxns.pkl'):
        logger.info('Start writing reactions to file: {}'.format(filename))

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
        logger.info('Start loading reactions from file: {}'.format(filename))
        d = pickle_load(filename)
        logger.info('Finish loading {} reactions'.format(len(d['reactions'])))

        return cls(d['molecules'], d['reactions'])

    def to_file_by_ids(self, filename='rxns.json'):
        logger.info('Start writing reactions by ids to file: {}'.format(filename))
        reaction_ids = []
        for i, r in enumerate(self.reactions):
            reaction_ids.append(r.as_dict())
        with open(filename, 'w') as f:
            json.dump(reaction_ids, f)

    @classmethod
    def from_file_by_ids(cls, filename, db_path):
        logger.info('Start loading reactions by ids from file: {}'.format(filename))

        db = DatabaseOperation.from_file(db_path)
        mols = db.to_molecules(purify=True)
        id_to_mol_map = {m.id: m for m in mols}

        with open(filename, 'r') as f:
            reactions = json.load(f)

        rxns = []
        for r in tqdm(reactions):
            reactants = [id_to_mol_map[i] for i in r['reactants']]
            products = [id_to_mol_map[i] for i in r['products']]
            broken_bond = r['broken_bond']
            rxn = Reaction(reactants, products, broken_bond)
            rxns.append(rxn)

        logger.info('Finish loading {} reactions'.format(len(reactions)))

        return cls(mols, rxns)
