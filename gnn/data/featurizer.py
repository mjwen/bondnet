"""
Featurize a molecule heterograph of atom, bond, and global nodes with RDkit.
"""
# pylint: disable=no-member,not-callable

import numpy as np
import torch
import os
import pickle
import warnings
from collections import defaultdict
import dgl
from dgl.data.chem.utils import mol_to_complete_graph

try:
    from rdkit import Chem
    from rdkit.Chem import rdmolops
    from rdkit.Chem import ChemicalFeatures
    from rdkit import RDConfig
except ImportError:
    pass


class AtomFeaturizer:
    """
    Featurization for all atoms in a molecule. The atom indices will be preserved.
    """

    def __init__(self, species):
        self.species = sorted(species)

    @property
    def feature_size(self):
        return self._feature_size

    def __call__(self, mol):
        """
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object

        Returns
        -------
        atom_feats_dict : dict
            Dictionary for atom features
        """
        mol = rdmolops.AddHs(mol, explicitOnly=True)

        atom_feats_dict = defaultdict(list)
        is_donor = defaultdict(int)
        is_acceptor = defaultdict(int)

        fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        mol_feats = mol_featurizer.GetFeaturesForMol(mol)

        for i in range(len(mol_feats)):
            if mol_feats[i].GetFamily() == "Donor":
                node_list = mol_feats[i].GetAtomIds()
                for u in node_list:
                    is_donor[u] = 1
            elif mol_feats[i].GetFamily() == "Acceptor":
                node_list = mol_feats[i].GetAtomIds()
                for u in node_list:
                    is_acceptor[u] = 1

        num_atoms = mol.GetNumAtoms()
        for u in range(num_atoms):
            atom = mol.GetAtomWithIdx(u)
            symbol = atom.GetSymbol()
            atom_type = atom.GetAtomicNum()
            aromatic = atom.GetIsAromatic()
            hybridization = atom.GetHybridization()
            num_h = atom.GetTotalNumHs()
            atom_feats_dict["node_type"].append(atom_type)

            h_u = []
            h_u += one_hot_encoding(symbol, self.species)
            h_u.append(atom_type)
            h_u.append(is_acceptor[u])
            h_u.append(is_donor[u])
            h_u.append(int(aromatic))
            h_u += one_hot_encoding(
                hybridization,
                [
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                ],
            )
            h_u.append(num_h)
            atom_feats_dict["a_feat"].append(h_u)

        atom_feats_dict["a_feat"] = torch.tensor(
            np.asarray(atom_feats_dict["a_feat"], dtype=np.float32)
        )
        atom_feats_dict["node_type"] = torch.tensor(
            np.asarray(atom_feats_dict["node_type"], dtype=np.int64)
        )

        self._feature_size = len(atom_feats_dict["a_feat"][0])

        return atom_feats_dict


class BondFeaturizer:
    """
    Featurization for all bonds in a molecule. The bond indices will be preserved.
    """

    def __init__(self):
        pass

    @property
    def feature_size(self):
        return self._feature_size

    def __call__(self, mol, self_loop=False):
        """
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object
        self_loop : bool
            Whether to add self loops. Default to be False.

        Returns
        -------
        bond_feats_dict : dict
            Dictionary for bond features
        """
        mol = rdmolops.AddHs(mol, explicitOnly=True)

        bond_feats_dict = defaultdict(list)

        num_bonds = mol.GetNumBonds()
        if num_bonds < 1:
            warnings.warn("molecular has no bonds")

        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            bond_type = bond.GetBondType()
            feature = one_hot_encoding(
                bond_type,
                [
                    Chem.rdchem.BondType.SINGLE,
                    Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE,
                    Chem.rdchem.BondType.AROMATIC,
                    Chem.rdchem.BondType.IONIC,
                    None,
                ],
            )
            bond_feats_dict["b_feat"].append(feature)

        bond_feats_dict["b_feat"] = torch.tensor(
            np.asarray(bond_feats_dict["b_feat"], dtype=np.float32)
        )

        self._feature_size = len(bond_feats_dict["b_feat"][0])

        return bond_feats_dict


class GlobalStateFeaturizer:
    """
    Featurization the global state of a molecules.  
    """

    @property
    def feature_size(self):
        return self._feature_size

    def __call__(self, charge):
        global_feats_dict = dict()
        g = one_hot_encoding(charge, [-1, 0, 1])
        self._feature_size = len(g)
        global_feats_dict["g_feat"] = torch.tensor([g])
        return global_feats_dict


class HeteroMoleculeGraph:
    """
    Convert a RDKit molecule to a DGLHeteroGraph and featurize for it.
    """

    def __init__(
        self,
        add_self_loop=False,
        atom_featurizer=None,
        bond_featurizer=None,
        global_state_featurizer=None,
    ):
        # TODO due to the way we are constructing the graph, self_loop seems not working
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.global_state_featurizer = global_state_featurizer

    def build_graph_and_featurize(self, mol, charge):
        """
        Build an a heterograph, with three types of nodes: atom, bond, and glboal
        state, and then featurize the graph.
        
        Args:
            mol (rdkit mol): a rdkit molecule
        
        Returns:
            g: dgl heterograph
            bond_idx_to_atom_idx (dict): mapping between two type bond indices, key is
                integer bond index, and value is a tuple of atom indices that specify
                the bond.
        """

        g = self.build_graph(mol)
        g = self.featurize(g, mol, charge)
        return g

    def build_graph(self, mol):
        num_atoms = mol.GetNumAtoms()

        # bonds
        num_bonds = mol.GetNumBonds()
        bond_idx_to_atom_idx = dict()
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            bond_idx_to_atom_idx[i] = (u, v)

        a2b = []
        b2a = []
        for a in range(num_atoms):
            for b, bond in bond_idx_to_atom_idx.items():
                if a in bond:
                    b2a.append([b, a])
                    a2b.append([a, b])

        a2g = [(a, 0) for a in range(num_atoms)]
        g2a = [(0, a) for a in range(num_atoms)]
        b2g = [(b, 0) for b in range(num_bonds)]
        g2b = [(0, b) for b in range(num_bonds)]

        g = dgl.heterograph(
            {
                ("atom", "a2b", "bond"): a2b,
                ("bond", "b2a", "atom"): b2a,
                ("atom", "a2g", "global"): a2g,
                ("global", "g2a", "atom"): g2a,
                ("bond", "b2g", "global"): b2g,
                ("global", "g2b", "bond"): g2b,
            }
        )

        return g

    def featurize(self, g, mol, charge):

        if self.atom_featurizer is not None:
            g.nodes["atom"].data.update(self.atom_featurizer(mol))
            g.nodes["bond"].data.update(self.bond_featurizer(mol))
            g.nodes["global"].data.update(self.global_state_featurizer(charge))

        return g

    @staticmethod
    def get_bond_to_atom_map(g):
        """
        Query which atoms are associated with the bonds.
        
        Args:
            g: dgl hetero graph
        
        Returns:
            dict: with bond index as the key and a tuple of atom indices of atoms that
                form the bond.
        """
        nbonds = g.number_of_nodes("bond")
        bond_to_atom_map = dict()
        for i in range(nbonds):
            atoms = g.successors(i, "b2a")
            bond_to_atom_map[i] = sorted(atoms)
        return bond_to_atom_map

    @staticmethod
    def get_atom_to_bond_map(g):
        """
        Query which bonds are associated with the atoms.
        
        Args:
            g: dgl hetero graph
        
        Returns:
            dict: with atom index as the key and a list of indices of bonds is
            connected to the atom.
        """
        natoms = g.number_of_nodes("atom")
        atom_to_bond_map = dict()
        for i in range(natoms):
            bonds = g.successors(i, "a2b")
            atom_to_bond_map[i] = sorted(list(bonds))
        return atom_to_bond_map


def one_hot_encoding(x, allowable_set):
    """One-hot encoding.

    Parameters
    ----------
    x : str, int or Chem.rdchem.HybridizationType
    allowable_set : list
        The elements of the allowable_set should be of the
        same type as x.

    Returns
    -------
    list
        List of boolean values where at most one value is True.
        If the i-th value is True, then we must have
        x == allowable_set[i].
    """
    return list(map(lambda s: x == s, allowable_set))
