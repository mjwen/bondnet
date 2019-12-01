"""
Featurize a molecule heterograph of atom, bond, and global nodes with RDkit.
"""
# pylint: disable=no-member,not-callable

import torch
import os
import warnings
from collections import defaultdict
import dgl

try:
    from rdkit import Chem
    from rdkit.Chem import rdmolops
    from rdkit.Chem import ChemicalFeatures
    from rdkit import RDConfig
except ImportError:
    pass


class BaseFeaturizer:
    def __init__(self, dtype="float32"):
        if dtype not in ["float32", "float64"]:
            raise ValueError(
                "`dtype` should be `float32` or `float64`, but got `{}`.".format(dtype)
            )
        self.dtype = dtype

    @property
    def feature_size(self):
        """
        Returns:
            an int of the feature size.
        """
        raise NotImplementedError

    @property
    def feature_name(self):
        """
        Returns:
            a list of the names of each feature. Should be of the same length as
            `feature_size`.
        """
        raise NotImplementedError

    def __call__(self, mol):
        """
        Returns:
            A dictionary of the features.
        """
        # TODO  we may want to change the return type to be a tensor instead of a dict
        raise NotImplementedError


class AtomFeaturizer(BaseFeaturizer):
    """
    Featurize atoms in a molecule. The atom indices will be preserved.
    """

    def __init__(self, species, dtype="float32"):
        super(AtomFeaturizer, self).__init__(dtype)
        self.species = sorted(species)
        self._feature_size = None
        self._feature_name = None

    @property
    def feature_size(self):
        return self._feature_size

    @property
    def feature_name(self):
        return self._feature_name

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
            feat = []
            feat.append(is_acceptor[u])
            feat.append(is_donor[u])

            atom = mol.GetAtomWithIdx(u)

            # feat.append(atom.GetDegree())
            feat.append(atom.GetTotalDegree())

            # feat.append(atom.GetExplicitValence())
            # feat.append(atom.GetImplicitValence())
            feat.append(atom.GetTotalValence())

            # feat.append(atom.GetFormalCharge())
            feat.append(atom.GetNumRadicalElectrons())

            feat.append(int(atom.GetIsAromatic()))
            feat.append(int(atom.IsInRing()))

            # feat.append(atom.GetNumExplicitHs())
            # feat.append(atom.GetNumImplicitHs())
            feat.append(atom.GetTotalNumHs())

            # feat.append(atom.GetAtomicNum())
            feat += one_hot_encoding(atom.GetSymbol(), self.species)

            feat += one_hot_encoding(
                atom.GetHybridization(),
                [
                    Chem.rdchem.HybridizationType.S,
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2,
                ],
            )

            atom_feats_dict["feat"].append(feat)

        dtype = getattr(torch, self.dtype)
        atom_feats_dict["feat"] = torch.tensor(atom_feats_dict["feat"], dtype=dtype)

        self._feature_size = len(atom_feats_dict["feat"][0])
        self._feature_name = (
            [
                "acceptor",
                "donor",
                # "degree",
                "total degree",
                # "explicit valence",
                # "implicit valence",
                "total valence",
                # "formal charge",
                "num radical electrons",
                "is aromatic",
                "is in ring",
                # "num explicit H",
                # "num implicit H",
                "num total H",
                # "atomic number",
            ]
            + ["chemical symbol"] * len(self.species)
            + ["hybridization"] * 6
        )

        return atom_feats_dict


class BondFeaturizer(BaseFeaturizer):
    """
    Featurize all bonds in a molecule. The bond indices will be preserved.
    """

    def __init__(self, dtype="float32"):
        super(BondFeaturizer, self).__init__(dtype)

    @property
    def feature_size(self):
        return self._feature_size

    @property
    def feature_name(self):
        return self._feature_name

    def __call__(self, mol):
        """
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object

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

        for u in range(num_bonds):
            bond = mol.GetBondWithIdx(u)

            feat = []

            feat.append(int(bond.GetIsAromatic()))
            feat.append(int(bond.IsInRing()))
            feat.append(int(bond.GetIsConjugated()))

            feat += one_hot_encoding(
                bond.GetBondType(),
                [
                    Chem.rdchem.BondType.SINGLE,
                    Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE,
                    # Chem.rdchem.BondType.AROMATIC,
                    # Chem.rdchem.BondType.IONIC,
                ],
            )
            bond_feats_dict["feat"].append(feat)

        dtype = getattr(torch, self.dtype)
        bond_feats_dict["feat"] = torch.tensor(bond_feats_dict["feat"], dtype=dtype)

        self._feature_size = len(bond_feats_dict["feat"][0])
        self._feature_name = ["is aromatic", "is in ring", "is conjugated"] + ["type"] * 3

        return bond_feats_dict


class GlobalStateFeaturizer(BaseFeaturizer):
    """
    Featurize the global state of a molecules.
    """

    def __init__(self, dtype="float32"):
        super(GlobalStateFeaturizer, self).__init__(dtype)

    @property
    def feature_size(self):
        return self._feature_size

    @property
    def feature_name(self):
        return self._feature_name

    def __call__(self, charge):
        global_feats_dict = dict()
        g = one_hot_encoding(charge, [-1, 0, 1])
        dtype = getattr(torch, self.dtype)
        global_feats_dict["feat"] = torch.tensor([g], dtype=dtype)

        self._feature_size = len(g)
        self._feature_name = ["charge"] * len(g)

        return global_feats_dict


class HeteroMoleculeGraph:
    """
    Convert a RDKit molecule to a DGLHeteroGraph and featurize for it.
    """

    def __init__(
        self,
        atom_featurizer=None,
        bond_featurizer=None,
        global_state_featurizer=None,
        self_loop=True,
    ):
        # TODO due to the way we are constructing the graph, self_loop seems not working
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.global_state_featurizer = global_state_featurizer
        self.self_loop = self_loop

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
        mol = rdmolops.AddHs(mol, explicitOnly=True)
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

        edges_dict = {
            ("atom", "a2b", "bond"): a2b,
            ("bond", "b2a", "atom"): b2a,
            ("atom", "a2g", "global"): a2g,
            ("global", "g2a", "atom"): g2a,
            ("bond", "b2g", "global"): b2g,
            ("global", "g2b", "bond"): g2b,
        }
        if self.self_loop:
            a2a = [(i, i) for i in range(num_atoms)]
            b2b = [(i, i) for i in range(num_bonds)]
            g2g = [(0, 0)]
            edges_dict.update(
                {
                    ("atom", "a2a", "atom"): a2a,
                    ("bond", "b2b", "bond"): b2b,
                    ("global", "g2g", "global"): g2g,
                }
            )
        g = dgl.heterograph(edges_dict)

        return g

    def featurize(self, g, mol, charge):

        if self.atom_featurizer is not None:
            g.nodes["atom"].data.update(self.atom_featurizer(mol))
        if self.bond_featurizer is not None:
            g.nodes["bond"].data.update(self.bond_featurizer(mol))
        if self.global_state_featurizer is not None:
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
        List of int (0 or 1) where at most one value is 1.
        If the i-th value is 1, then we must have x == allowable_set[i].
    """
    return list(map(int, list(map(lambda s: x == s, allowable_set))))

