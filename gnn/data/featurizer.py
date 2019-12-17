"""
Featurize a molecule heterograph of atom, bond, and global nodes with RDkit.
"""

import torch
import os
import warnings
from collections import defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit.Chem.rdchem import GetPeriodicTable


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

    def __call__(self, mol, **kwargs):
        """
        Returns:
            A dictionary of the features.
        """
        raise NotImplementedError


class AtomFeaturizer(BaseFeaturizer):
    """
    Featurize atoms in a molecule.
    The atom indices will be preserved, i.e. feature i corresponds to atom i.
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

    def __call__(self, mol, **kwargs):
        """
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object

        Returns
        -------
            Dictionary for atom features
        """
        feats = []
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
            ft = [is_acceptor[u], is_donor[u]]

            atom = mol.GetAtomWithIdx(u)

            # feat.append(atom.GetDegree())
            ft.append(atom.GetTotalDegree())

            # feat.append(atom.GetExplicitValence())
            # feat.append(atom.GetImplicitValence())
            ft.append(atom.GetTotalValence())

            # feat.append(atom.GetFormalCharge())
            ft.append(atom.GetNumRadicalElectrons())

            ft.append(int(atom.GetIsAromatic()))
            ft.append(int(atom.IsInRing()))

            # feat.append(atom.GetNumExplicitHs())
            # feat.append(atom.GetNumImplicitHs())
            ft.append(atom.GetTotalNumHs())

            # feat.append(atom.GetAtomicNum())
            ft += one_hot_encoding(atom.GetSymbol(), self.species)

            ft += one_hot_encoding(
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

            feats.append(ft)

        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
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

        return {"feat": feats}


class BondAsNodeFeaturizer(BaseFeaturizer):
    """
    Featurize all bonds in a molecule.

    The bond indices will be preserved, i.e. feature i corresponds to atom i.
    The number of features will be equal to the number of bonds in the molecule,
    so this is suitable for the case where we represent bond as graph nodes.

    See Also:
        BondAsEdgeBidirectedFeaturizer
    """

    def __init__(self, length_featurizer=None, dtype="float32"):
        super(BondAsNodeFeaturizer, self).__init__(dtype)
        self._feature_size = None
        self._feature_name = None

        if length_featurizer == "bin":
            self.length_featurizer = DistanceBins(low=2.0, high=6.0, num_bins=10)
        elif length_featurizer == "rfb":
            self.length_featurizer = RBF(low=2.0, high=6.0, gap=0.1, dim=1)
        else:
            self.length_featurizer = None

    @property
    def feature_size(self):
        return self._feature_size

    @property
    def feature_name(self):
        return self._feature_name

    def __call__(self, mol, **kwargs):
        """
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object

        Returns
        -------
            Dictionary for bond features
        """
        feats = []

        num_bonds = mol.GetNumBonds()
        if num_bonds < 1:
            warnings.warn("molecular has no bonds")

        for u in range(num_bonds):
            bond = mol.GetBondWithIdx(u)

            ft = [
                int(bond.GetIsAromatic()),
                int(bond.IsInRing()),
                int(bond.GetIsConjugated()),
            ]

            ft += one_hot_encoding(
                bond.GetBondType(),
                [
                    Chem.rdchem.BondType.SINGLE,
                    Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE,
                    # Chem.rdchem.BondType.AROMATIC,
                    # Chem.rdchem.BondType.IONIC,
                ],
            )

            if self.length_featurizer:
                at1 = bond.GetBeginAtomIdx()
                at2 = bond.GetEndAtomIdx()
                atoms_pos = mol.GetConformer().GetPositions()
                bond_length = np.linalg.norm(atoms_pos[at1] - atoms_pos[at2])
                ft += self.length_featurizer(bond_length)

            feats.append(ft)

        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = ["is aromatic", "is in ring", "is conjugated"] + ["type"] * 3
        if self.length_featurizer:
            self._feature_name += self.length_featurizer.feature_name

        return {"feat": feats}


class BondAsEdgeBidirectedFeaturizer(BaseFeaturizer):
    """
    Featurize all bonds in a molecule.

    Feature of bond 0 is assigned to graph edges 0 and 1, feature of bond 1 is assigned
    to graph edges 2, and 3 ... If `self_loop` is `True`, graph edge 2Nb to 2Nb+Na-1
    will also have features, but they are not determined from the actual bond in the
    molecule.

    This is suitable for the case where we represent bond as edges of bidirected graph.
    For example, it can be used together :meth:`gnn.data.grapher.HomoBidirectedGraph`.

    Args:
        self_loop (bool): whether to let the each node connect to itself
        length_featurizer (str): method to featurize bond length, options are `bin` and
        `rbf`, if `None` bond length feature is not used._

    See Also:
        BondAsNodeFeaturizer
        BondAsEdgeCompleteFeaturizer
    """

    def __init__(self, self_loop=True, length_featurizer=None, dtype="float32"):
        super(BondAsEdgeBidirectedFeaturizer, self).__init__(dtype)
        self.self_loop = self_loop
        self._feature_size = None
        self._feature_name = None

        if length_featurizer == "bin":
            self.length_featurizer = DistanceBins(low=2.0, high=6.0, num_bins=10)
        elif length_featurizer == "rfb":
            self.length_featurizer = RBF(low=2.0, high=6.0, gap=0.1, dim=1)
        else:
            self.length_featurizer = None

    @property
    def feature_size(self):
        return self._feature_size

    @property
    def feature_name(self):
        return self._feature_name

    def __call__(self, mol, **kwargs):
        """
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object

        Returns
        -------
            Dictionary for bond features
        """
        feats = []
        num_bonds = mol.GetNumBonds()
        if num_bonds < 1:
            warnings.warn("molecular has no bonds")

        for u in range(num_bonds):
            bond = mol.GetBondWithIdx(u)

            ft = [
                int(bond.GetIsAromatic()),
                int(bond.IsInRing()),
                int(bond.GetIsConjugated()),
            ]

            ft += one_hot_encoding(
                bond.GetBondType(),
                [
                    Chem.rdchem.BondType.SINGLE,
                    Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE,
                    # Chem.rdchem.BondType.AROMATIC,
                    # Chem.rdchem.BondType.IONIC,
                    None,
                ],
            )

            if self.length_featurizer:
                at1 = bond.GetBeginAtomIdx()
                at2 = bond.GetEndAtomIdx()
                atoms_pos = mol.GetConformer().GetPositions()
                bond_length = np.linalg.norm(atoms_pos[at1] - atoms_pos[at2])
                ft += self.length_featurizer(bond_length)

            feats.extend([ft, ft])

        if self.self_loop:
            for i in range(mol.GetNumAtoms()):

                # use -1 to denote not applicable, not ideal but acceptable
                ft = [-1, -1, -1]

                # no bond type for self loop
                ft += one_hot_encoding(
                    None,
                    [
                        Chem.rdchem.BondType.SINGLE,
                        Chem.rdchem.BondType.DOUBLE,
                        Chem.rdchem.BondType.TRIPLE,
                        # Chem.rdchem.BondType.AROMATIC,
                        # Chem.rdchem.BondType.IONIC,
                        None,
                    ],
                )

                # bond distance
                if self.length_featurizer:
                    bond_length = 0.0
                    ft += self.length_featurizer(bond_length)

                feats.append(ft)

        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = ["is aromatic", "is in ring", "is conjugated"] + ["type"] * 4
        if self.length_featurizer:
            self._feature_name += self.length_featurizer.feature_name

        return {"feat": feats}


class BondAsEdgeCompleteFeaturizer(BaseFeaturizer):
    """
    Featurize all bonds in a molecule.

    Create features between atom pairs (0, 0), (0,1), (0,2), ... (1,0), (1,1), (1,2), ...
    If not `self_loop`, (0,0), (1,1) ... pairs will not be present.

    This is suitable for the case where we represent bond as complete graph edges. For
    example, it can be used together :meth:`gnn.data.grapher.HomoCompleteGraph`.

    See Also:
        BondAsNodeFeaturizer
        BondAsEdgeBidirectedFeaturizer
    """

    def __init__(self, self_loop=True, length_featurizer=None, dtype="float32"):
        super(BondAsEdgeCompleteFeaturizer, self).__init__(dtype)
        self.self_loop = self_loop
        self._feature_size = None
        self._feature_name = None

        if length_featurizer == "bin":
            self.length_featurizer = DistanceBins(low=2.0, high=6.0, num_bins=10)
        elif length_featurizer == "rfb":
            self.length_featurizer = RBF(low=2.0, high=6.0, gap=0.1, dim=1)
        else:
            self.length_featurizer = None

    @property
    def feature_size(self):
        return self._feature_size

    @property
    def feature_name(self):
        return self._feature_name

    def __call__(self, mol, **kwargs):
        """
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object

        Returns
        -------
            Dictionary for bond features
        """
        feats = []

        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        if num_bonds < 1:
            warnings.warn("molecular has no bonds")

        for u in range(num_atoms):
            for v in range(num_atoms):
                if u == v and not self.self_loop:
                    continue

                bond = mol.GetBondBetweenAtoms(u, v)
                if bond is None:
                    bond_type = None
                    ft = [-1, -1, -1]
                else:
                    bond_type = bond.GetBondType()
                    ft = [
                        int(bond.GetIsAromatic()),
                        int(bond.IsInRing()),
                        int(bond.GetIsConjugated()),
                    ]

                ft += one_hot_encoding(
                    bond_type,
                    [
                        Chem.rdchem.BondType.SINGLE,
                        Chem.rdchem.BondType.DOUBLE,
                        Chem.rdchem.BondType.TRIPLE,
                        # Chem.rdchem.BondType.AROMATIC,
                        # Chem.rdchem.BondType.IONIC,
                        None,
                    ],
                )

                # bond distance
                if self.length_featurizer:
                    atoms_pos = mol.GetConformer().GetPositions()
                    bond_length = np.linalg.norm(atoms_pos[u] - atoms_pos[v])
                    ft += self.length_featurizer(bond_length)

                feats.append(ft)

        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = ["is aromatic", "is in ring", "is conjugated"] + ["type"] * 4
        if self.length_featurizer:
            self._feature_name += self.length_featurizer.feature_name

        return {"feat": feats}


class MolChargeFeaturizer(BaseFeaturizer):
    """
    Featurize the global state of a molecules using charge.
    """

    def __init__(self, dtype="float32"):
        super(MolChargeFeaturizer, self).__init__(dtype)
        self._feature_size = None
        self._feature_name = None

    @property
    def feature_size(self):
        return self._feature_size

    @property
    def feature_name(self):
        return self._feature_name

    def __call__(self, mol, **kwargs):
        try:
            charge = kwargs["charge"]
        except KeyError as e:
            raise KeyError("{} charge needed for {}.".format(e, self.__class__.__name__))

        g = one_hot_encoding(charge, [-1, 0, 1])

        feats = torch.tensor([g], dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = ["charge"] * feats.shape[1]

        return {"feat": feats}


class MolWeightFeaturizer(BaseFeaturizer):
    """
    Featurize the global state of a molecules using number of atoms, number of bonds,
    and its weight.
    """

    def __init__(self, dtype="float32"):
        super(MolWeightFeaturizer, self).__init__(dtype)
        self._feature_size = None
        self._feature_name = None

    @property
    def feature_size(self):
        return self._feature_size

    @property
    def feature_name(self):
        return self._feature_name

    def __call__(self, mol, **kwargs):

        pd = GetPeriodicTable()
        g = [
            mol.GetNumAtoms(),
            mol.GetNumBonds(),
            sum([pd.GetAtomicWeight(a.GetAtomicNum()) for a in mol.GetAtoms()]),
        ]

        feats = torch.tensor([g], dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = ["num atoms", "num bonds", "molecule weight"]

        return {"feat": feats}


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


class DistanceBins(BaseFeaturizer):
    """
    Put the distance into a bins. As used in MPNN.

    Args:
        low (float): lower bound of bin. Values smaller than this will all be put in
            the same bin.
        high (float): upper bound of bin. Values larger than this will all be put in
            the same bin.
        num_bins (int): number of bins. Besides two bins (one smaller than `low` and
            one larger than `high`) a number of `num_bins -2` bins will be evenly
            created between [low, high).

    """

    def __init__(self, low=2.0, high=6.0, num_bins=10):
        self.num_bins = num_bins
        self.bins = np.linspace(low, high, num_bins - 1, endpoint=True)
        self.bin_indices = np.arange(num_bins)

    @property
    def feature_size(self):
        return self.num_bins

    @property
    def feature_name(self):
        return ["dist bins"] * self.feature_size

    def __call__(self, distance):
        v = np.digitize(distance, self.bins)
        return one_hot_encoding(v, self.bin_indices)


class RBF(BaseFeaturizer):
    """
    Radial basis functions.
    e(d) = exp(- gamma * ||d - mu_k||^2)
    With the default parameters below, we are using a default settings:
    * gamma = 10
    * 0 <= mu_k <= 30 for k=1~300
    Parameters
    ----------
    low : float
        Smallest value to take for mu_k, default to be 0.
    high : float
        Largest value to take for mu_k, default to be 30.
    gap : float
        Difference between two consecutive values for mu_k, default to be 0.1.
    dim : int
        Output size for each center, default to be 1.
    """

    # TODO to really implement this
    def __init__(self, low=0.0, high=30.0, gap=0.1, dim=1):
        super(RBF, self).__init__()

        self._low = low
        self._high = high
        self._dim = dim

        self._n_centers = int(np.ceil((high - low) / gap))
        centers = np.linspace(low, high, self._n_centers)
        self.centers = torch.tensor(centers, dtype=torch.float, requires_grad=False)
        # self.centers = nn.Parameter(self.centers, requires_grad=False)
        self._fan_out = self._dim * self._n_centers
        self._gap = centers[1] - centers[0]

    def __call__(self, edge_distances):
        """
        Parameters
        ----------
        edge_distances : float32 tensor of shape (B, 1)
            Edge distances, B for the number of edges.
        Returns
        -------
        float32 tensor of shape (B, self._fan_out)
            Computed RBF results
        """
        radial = edge_distances - self.centers
        coef = -1 / self._gap
        return torch.exp(coef * (radial ** 2))
