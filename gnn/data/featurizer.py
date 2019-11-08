import numpy as np
import os
from collections import defaultdict
import torch
from functools import partial
import dgl
import dgl.backend as F

try:
    from rdkit import Chem
    from rdkit.Chem import rdmolfiles, rdmolops
except ImportError:
    pass


class AtomFeaturizer:
    """
    Featurization for all atoms in a molecule. The atom indices will be preserved.
    """

    def __init__(self, species):
        self.species = species

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
        mol_conformers = mol.GetConformers()
        assert len(mol_conformers) == 1

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
            h_u += [int(symbol == x) for x in self.species]
            h_u.append(atom_type)
            h_u.append(is_acceptor[u])
            h_u.append(is_donor[u])
            h_u.append(int(aromatic))
            h_u += [
                int(hybridization == x)
                for x in (
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                )
            ]
            h_u.append(num_h)
            atom_feats_dict["n_feat"].append(
                torch.tensor(np.array(h_u).astype(np.float32))
            )

        self._feature_size = len(atom_feats_dict["n_feat"][0])

        atom_feats_dict["n_feat"] = torch.stack(atom_feats_dict["n_feat"], dim=0)
        atom_feats_dict["node_type"] = torch.tensor(
            np.array(atom_feats_dict["node_type"]).astype(np.int64)
        )

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

        mol_conformers = mol.GetConformers()
        assert len(mol_conformers) == 1
        geom = mol_conformers[0].GetPositions()

        num_atoms = mol.GetNumAtoms()
        assert (
            num_atoms > 1
        ), "number of atoms < 2; cannot featurize bond edge of molecule"

        for u in range(num_atoms):
            for v in range(num_atoms):
                if u == v and not self_loop:
                    continue

                e_uv = mol.GetBondBetweenAtoms(u, v)
                if e_uv is None:
                    bond_type = None
                else:
                    bond_type = e_uv.GetBondType()
                bond_feats_dict["e_feat"].append(
                    [
                        float(bond_type == x)
                        for x in (
                            Chem.rdchem.BondType.SINGLE,
                            Chem.rdchem.BondType.DOUBLE,
                            Chem.rdchem.BondType.TRIPLE,
                            Chem.rdchem.BondType.AROMATIC,
                            None,
                        )
                    ]
                )
                bond_feats_dict["distance"].append(np.linalg.norm(geom[u] - geom[v]))

        bond_feats_dict["e_feat"] = torch.tensor(
            np.array(bond_feats_dict["e_feat"]).astype(np.float32)
        )

        self._feature_size = len(bond_feats_dict["e_feat"][0])

        bond_feats_dict["distance"] = torch.tensor(
            np.array(bond_feats_dict["distance"]).astype(np.float32)
        ).reshape(-1, 1)

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
        global_feats_dict["g_feat"] = torch.tensor(g)
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

        g, bond_idx_to_atom_idx = self.build_graph(mol)
        g = self.featurize(g, mol, charge)
        return g, bond_idx_to_atom_idx

    def build_graph(self, mol):
        num_atoms = mol.GetNumAtoms()

        # bonds
        num_bonds = mol.GetNumBonds()
        bond_idx_to_atom_idx = dict()
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            bond_idx_to_atom_idx[i] = sorted(tuple([u, v]))

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
                ("atom", "anb", "bond"): a2b,
                ("bond", "anb", "atom"): b2a,
                ("atom", "ang", "global"): a2g,
                ("global", "ang", "atom"): g2a,
                ("bond", "bng", "global"): b2g,
                ("global", "bng", "bond"): g2b,
            }
        )
        return g, bond_idx_to_atom_idx

    def featurize(self, g, mol, charge):

        if self.atom_featurizer is not None:
            g.nodes["atom"].data.update(self.atom_featurizer(mol))
            g.nodes["bond"].data.update(self.bond_featurizer(mol))
            g.nodes["global"].data.update(self.global_state_featurizer(charge))

        return g


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

