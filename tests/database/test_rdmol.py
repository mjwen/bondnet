import numpy as np
from rdkit.Chem import BondType
from gnn.core.rdmol import (
    remove_metals,
    create_rdkit_mol,
    create_rdkit_mol_from_mol_graph,
    fragment_rdkit_mol,
)

from .utils import create_LiEC_pymatgen_mol, create_LiEC_mol_graph, create_LiEC_rdkit_mol


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

    rd_species = [a.GetSymbol() for a in m.GetAtoms()]
    rd_coords = m.GetConformer().GetPositions()
    rd_bonds = dict()
    for bond in m.GetBonds():
        idx = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        tp = bond.GetBondType()
        rd_bonds[idx] = tp

    assert species == rd_species
    assert np.allclose(coords, rd_coords)
    assert bond_types == rd_bonds


def test_create_rdkit_mol_from_mol_graph():
    mol_graph = create_LiEC_mol_graph()

    pymatgen_mol = mol_graph.molecule
    species = [str(s) for s in pymatgen_mol.species]
    coords = pymatgen_mol.cart_coords

    m, bond_types = create_rdkit_mol_from_mol_graph(mol_graph)

    rd_species = [a.GetSymbol() for a in m.GetAtoms()]
    rd_coords = m.GetConformer().GetPositions()
    rd_bonds = dict()
    for bond in m.GetBonds():
        idx = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        tp = bond.GetBondType()
        rd_bonds[idx] = tp

    assert species == rd_species
    assert np.allclose(coords, rd_coords)
    assert bond_types == rd_bonds


def test_fragment_rdkit_mol():

    rdm = create_LiEC_rdkit_mol()
    # print("#" * 80, "original", Chem.MolToMolBlock(rdm, forceV3000=True))

    # break C C bond in the ring
    bond = (2, 5)
    mols, _ = fragment_rdkit_mol(rdm, bond)
    assert len(mols) == 1
    assert len(mols[0].GetBonds()) == 10

    # print("#" * 80, "C-C", Chem.MolToMolBlock(mols[0], forceV3000=True))

    # break the Li O bonds
    bond = (3, 6)
    mols, _ = fragment_rdkit_mol(rdm, bond)
    assert len(mols) == 2
    assert len(mols[0].GetBonds()) == 10 and len(mols[1].GetBonds()) == 0

    # print("#" * 80, "Li O", Chem.MolToMolBlock(mols[0], forceV3000=True))
    # print("#" * 80, "Li O", Chem.MolToMolBlock(mols[1], forceV3000=True))

    # # break C=O bond, generating two fragments
    bond = (3, 1)
    mols, _ = fragment_rdkit_mol(rdm, bond)
    assert len(mols) == 2
    assert len(mols[0].GetBonds()) == 9 and len(mols[1].GetBonds()) == 1

    # print("#" * 80, "C O", Chem.MolToMolBlock(mols[0], forceV3000=True))
    # print("#" * 80, "C O", Chem.MolToMolBlock(mols[1], forceV3000=True))
