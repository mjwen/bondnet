import pickle
from atomate.qchem.database import QChemCalcDb
import pymatgen
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.analysis.fragmenter import metal_edge_extender
from pymatgen.io.babel import BabelMolAdaptor
import openbabel as ob
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from gnn.data.utils import create_directory


def query_database(db_file="/Users/mjwen/Applications/mongo_db_access/sam_db.json"):
    # Create a json file that contains the credentials
    mmdb = QChemCalcDb.from_db_file(db_file, admin=True)

    # This contains all the production jobs(wb97xv/def2-tzvppd/smd(LiEC parameters)).
    # Every target_entries[i] is a dictionary of all the information for one job.
    target_entries = list(mmdb.collection.find({"tags.class": 'smd_production'}))
    return target_entries


def dump_database(database, size=None, fname='database.pkl'):
    size = size or 10
    database = database[:size]
    with open(fname, 'wb') as fout:
        pickle.dump(database, fout)


def recover_database(fname='database.pkl'):
    with open(fname, 'rb') as fin:
        database = pickle.load(fin)
    return database


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


class Molecule:
    def __init__(self, db_entry):
        self.db_entry = db_entry
        self.mol = pymatgen.Molecule.from_dict(db_entry['output']['optimized_molecule'])

        mol_graph = MoleculeGraph.with_local_env_strategy(
            self.mol, OpenBabelNN(order=True), reorder=False, extend_structure=False
        )
        self.mol_graph = metal_edge_extender(mol_graph)
        self.ob_adaptor = BabelMolAdaptor2.from_molecule_graph(self.mol_graph)

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
        f = self.mol.composition.alphabetical_formula
        return f.replace(' ', '')

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

    def write(self, filename=None, file_format='sdf'):
        return self.ob_adaptor.pybel_mol.write(file_format, filename, overwrite=True)

    def draw(self, filename=None, draw_2D=True):
        sdf = self.write(file_format='sdf')
        m = Chem.MolFromMolBlock(sdf)
        if draw_2D:
            AllChem.Compute2DCoords(m)

        filename = filename or 'mol.svg'
        filename = create_directory(filename)
        Draw.MolToFile(m, filename)


def create_dataset(db_path):
    db_entries = recover_database(db_path)

    with open('electrolyte.sdf', 'w') as fx, open('electrolyte.csv', 'w') as fy:
        for entry in db_entries:
            m = Molecule(entry)

            conn = m.get_connectivity()
            species = m.get_species()
            coords = m.get_coords()
            bond_order = m.get_bond_order()
            # print('conn', conn)
            # print('speices', species)
            # print('coords', coords)
            # print('bond_order', bond_order)

            # fname = 'images/' + formula + '.svg'
            # m.draw(fname)

            sdf = m.write(file_format='sdf')
            smiles = m.write(file_format='smi')
            formula = m.formula
            # print('smiles', smiles, 'formula', formula, '\nsdf', sdf)

            fx.write(sdf)


if __name__ == '__main__':
    db_path = '/Users/mjwen/Applications/mongo_db_access/database_small.pkl'
    create_dataset(db_path)
