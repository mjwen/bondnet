from rdkit import Chem
from rdkit.Chem import BondType
from gnn.database.molwrapper import (
    remove_metals,
    create_rdkit_mol,
    create_rdkit_mol_from_mol_graph,
)
from .utils import create_C2H4O1, create_LiEC_pymatgen_mol, create_LiEC_mol_graph


def test_isomorphic_bonds():
    # {0:3} first because products are ordered
    ref_identical = [{(0, 2), (0, 3)}, {(1, 2), (3, 4), (2, 5), (3, 6)}]

    mol = create_C2H4O1()
    iso_identical = [set(g) for g in mol.isomorphic_bonds]

    for v in iso_identical:
        assert v in ref_identical
    for v in ref_identical:
        assert v in iso_identical


def test_remove_metals():

    mol = create_LiEC_pymatgen_mol()
    mol = remove_metals(mol)
    assert len(mol) == 10
    assert mol.charge == -1


def test_create_rdkit_mol():

    # LiEC
    species = ["O", "C", "C", "O", "O", "C", "Li", "H", "H", "H", "H"]
    coords = [
        [0.3103, -1.1776, -0.3722],
        [-0.6822, -0.5086, 0.3490],
        [1.5289, -0.4938, -0.0925],
        [-1.9018, -0.6327, -0.0141],
        [-0.2475, 0.9112, 0.3711],
        [1.1084, 0.9722, -0.0814],
        [-2.0519, 1.1814, -0.2310],
        [2.2514, -0.7288, -0.8736],
        [1.9228, -0.8043, 0.8819],
        [1.1406, 1.4103, -1.0835],
        [1.7022, 1.5801, 0.6038],
    ]
    bond_types = {
        (0, 2): BondType.SINGLE,
        (0, 1): BondType.SINGLE,
        (2, 5): BondType.SINGLE,
        (2, 8): BondType.SINGLE,
        (4, 1): BondType.SINGLE,
        (5, 4): BondType.SINGLE,
        (5, 10): BondType.SINGLE,
        (7, 2): BondType.SINGLE,
        (9, 5): BondType.SINGLE,
        (4, 6): BondType.DATIVE,
        (3, 6): BondType.DATIVE,
        (3, 1): BondType.DOUBLE,
    }

    formal_charge = [0 for _ in range(len(species))]
    formal_charge[6] = -1  # set Li to -1 because of dative bond

    m = create_rdkit_mol(species, coords, bond_types, formal_charge)
    print(Chem.MolToMolBlock(m))


def test_create_rdkit_mol_from_mol_graph():
    mol_graph = create_LiEC_mol_graph()
    m = create_rdkit_mol_from_mol_graph(mol_graph)
    print(Chem.MolToMolBlock(m))
