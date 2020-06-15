"""
Featurize a molecule heterograph of atom, bond, and global nodes with RDkit.
"""

import torch
import os
import warnings
import itertools
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
        self._feature_size = None
        self._feature_name = None

    @property
    def feature_size(self):
        """
        Returns:
            an int of the feature size.
        """
        return self._feature_size

    @property
    def feature_name(self):
        """
        Returns:
            a list of the names of each feature. Should be of the same length as
            `feature_size`.
        """

        return self._feature_name

    def __call__(self, mol, **kwargs):
        """
        Returns:
            A dictionary of the features.
        """
        raise NotImplementedError


class BondFeaturizer(BaseFeaturizer):
    """
    Base featurize all bonds in a molecule.

    The bond indices will be preserved, i.e. feature i corresponds to atom i.
    The number of features will be equal to the number of bonds in the molecule,
    so this is suitable for the case where we represent bond as graph nodes.

    Args:
        length_featurizer (str or None): the featurizer for bond length.
        length_featurizer_args (dict): a dictionary of the arguments for the featurizer.
            If `None`, default values will be used, but typically not good because this
            should be specific to the dataset being used.
    """

    def __init__(
        self, length_featurizer=None, length_featurizer_args=None, dtype="float32"
    ):
        super(BondFeaturizer, self).__init__(dtype)
        self._feature_size = None
        self._feature_name = None

        if length_featurizer == "bin":
            if length_featurizer_args is None:
                length_featurizer_args = {"low": 0.0, "high": 2.5, "num_bins": 10}
            self.length_featurizer = DistanceBins(**length_featurizer_args)
        elif length_featurizer == "rbf":
            if length_featurizer_args is None:
                length_featurizer_args = {"low": 0.0, "high": 2.5, "num_centers": 10}
            self.length_featurizer = RBF(**length_featurizer_args)
        elif length_featurizer is None:
            self.length_featurizer = None
        else:
            raise ValueError(
                "Unsupported bond length featurizer: {}".format(length_featurizer)
            )


class BondAsNodeFeaturizerMinimum(BondFeaturizer):
    """
    Featurize all bonds in a molecule.

    Do not use bond type info.

    See Also:
        BondAsEdgeBidirectedFeaturizer
    """

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

        # Note, this needs to be set such that single atom molecule works
        num_feats = 7

        num_bonds = mol.GetNumBonds()

        if num_bonds == 0:
            ft = [0.0 for _ in range(num_feats)]
            if self.length_featurizer:
                ft += [0.0 for _ in range(len(self.length_featurizer.feature_name))]
            feats = [ft]

        else:

            ring = mol.GetRingInfo()
            allowed_ring_size = [3, 4, 5, 6, 7]

            feats = []
            for u in range(num_bonds):
                bond = mol.GetBondWithIdx(u)

                ft = [
                    int(bond.IsInRing()),
                ]

                for s in allowed_ring_size:
                    ft.append(ring.IsBondInRingOfSize(u, s))

                ft.append(int(bond.GetBondType() == Chem.rdchem.BondType.DATIVE))

                if self.length_featurizer:
                    at1 = bond.GetBeginAtomIdx()
                    at2 = bond.GetEndAtomIdx()
                    atoms_pos = mol.GetConformer().GetPositions()
                    bond_length = np.linalg.norm(atoms_pos[at1] - atoms_pos[at2])
                    ft += self.length_featurizer(bond_length)

                feats.append(ft)

        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = ["in_ring"] + ["ring size"] * 5 + ["dative"]
        if self.length_featurizer:
            self._feature_name += self.length_featurizer.feature_name

        return {"feat": feats}


class BondAsNodeFeaturizerFull(BondFeaturizer):
    """
    Featurize all bonds in a molecule.

    The bond indices will be preserved, i.e. feature i corresponds to atom i.
    The number of features will be equal to the number of bonds in the molecule,
    so this is suitable for the case where we represent bond as graph nodes.

    See Also:
        BondAsEdgeBidirectedFeaturizer
    """

    def __init__(
        self,
        length_featurizer=None,
        length_featurizer_args=None,
        dative=False,
        dtype="float32",
    ):
        super(BondAsNodeFeaturizerFull, self).__init__(
            length_featurizer, length_featurizer_args, dtype
        )
        self.dative = dative

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

        # Note, this needs to be set such that single atom molecule works
        if self.dative:
            num_feats = 12
        else:
            num_feats = 11

        num_bonds = mol.GetNumBonds()

        if num_bonds == 0:
            ft = [0.0 for _ in range(num_feats)]
            if self.length_featurizer:
                ft += [0.0 for _ in range(len(self.length_featurizer.feature_name))]
            feats = [ft]

        else:
            ring = mol.GetRingInfo()
            allowed_ring_size = [3, 4, 5, 6, 7]

            feats = []
            for u in range(num_bonds):
                bond = mol.GetBondWithIdx(u)

                ft = [
                    int(bond.IsInRing()),
                    int(bond.GetIsConjugated()),
                ]
                for s in allowed_ring_size:
                    ft.append(ring.IsBondInRingOfSize(u, s))

                allowed_bond_type = [
                    Chem.rdchem.BondType.SINGLE,
                    Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE,
                    Chem.rdchem.BondType.AROMATIC,
                    # Chem.rdchem.BondType.IONIC,
                ]
                if self.dative:
                    allowed_bond_type.append(Chem.rdchem.BondType.DATIVE)
                ft += one_hot_encoding(bond.GetBondType(), allowed_bond_type)

                if self.length_featurizer:
                    at1 = bond.GetBeginAtomIdx()
                    at2 = bond.GetEndAtomIdx()
                    atoms_pos = mol.GetConformer().GetPositions()
                    bond_length = np.linalg.norm(atoms_pos[at1] - atoms_pos[at2])
                    ft += self.length_featurizer(bond_length)

                feats.append(ft)

        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = (
            ["in_ring", "conjugated"]
            + ["ring size"] * 5
            + ["single", "double", "triple", "aromatic"]
        )
        if self.dative:
            self._feature_name += ["dative"]
        if self.length_featurizer:
            self._feature_name += self.length_featurizer.feature_name

        return {"feat": feats}


class BondAsNodeCompleteFeaturizer(BondFeaturizer):
    """
    Featurize all bonds in a molecule.

    Bonds is different from the typical notion. Here we assume there is a bond between
    every atom pairs.

    The order of the bonds are (0,1), (0,2), ... , (0, N-1), (1,2), (1,3), ...,
    (N-2, N-1), where N is the number of atoms.
    """

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
        if not self.length_featurizer:
            raise ValueError(
                f"length_featurizer (None) needs to be provided for "
                f"{self.__class__.__name__}, which is the only feature."
            )

        num_atoms = mol.GetNumAtoms()

        feats = []
        for u, v in itertools.combinations(range(num_atoms), 2):
            atoms_pos = mol.GetConformer().GetPositions()
            bond_length = np.linalg.norm(atoms_pos[u] - atoms_pos[v])
            ft = self.length_featurizer(bond_length)
            feats.append(ft)

        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = self.length_featurizer.feature_name

        return {"feat": feats}


class BondAsEdgeBidirectedFeaturizer(BondFeaturizer):
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

    def __init__(
        self,
        self_loop=True,
        length_featurizer=None,
        length_featurizer_args=None,
        dtype="float32",
    ):
        self.self_loop = self_loop
        super(BondAsEdgeBidirectedFeaturizer, self).__init__(
            length_featurizer, length_featurizer_args, dtype
        )

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


class BondAsEdgeCompleteFeaturizer(BondFeaturizer):
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

    def __init__(
        self,
        self_loop=True,
        length_featurizer=None,
        length_featurizer_args=None,
        dtype="float32",
    ):
        self.self_loop = self_loop
        super(BondAsEdgeCompleteFeaturizer, self).__init__(
            length_featurizer, length_featurizer_args, dtype
        )

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


class AtomFeaturizerMinimum(BaseFeaturizer):
    """
    Featurize atoms in a molecule.

    Mimimum set of info without hybridization info.
    """

    def __call__(self, mol, **kwargs):
        """
        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object

            Also `extra_feats_info` should be provided as `kwargs` as additional info.

        Returns:
            Dictionary of atom features
        """
        try:
            species = sorted(kwargs["dataset_species"])
        except KeyError as e:
            raise KeyError(
                "{} `dataset_species` needed for {}.".format(e, self.__class__.__name__)
            )
        try:
            feats_info = kwargs["extra_feats_info"]
        except KeyError as e:
            raise KeyError(
                "{} `extra_feats_info` needed for {}.".format(e, self.__class__.__name__)
            )

        feats = []

        ring = mol.GetRingInfo()
        allowed_ring_size = [3, 4, 5, 6, 7]
        num_atoms = mol.GetNumAtoms()
        for i in range(num_atoms):
            ft = []
            atom = mol.GetAtomWithIdx(i)

            ft.append(atom.GetTotalDegree())
            ft.append(int(atom.IsInRing()))
            ft.append(atom.GetTotalNumHs(includeNeighbors=True))

            ft += one_hot_encoding(atom.GetSymbol(), species)

            for s in allowed_ring_size:
                ft.append(ring.IsAtomInRingOfSize(i, s))

            feats.append(ft)

        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = (
            ["total degree", "is in ring", "total H"]
            + ["chemical symbol"] * len(species)
            + ["ring size"] * 5
        )

        return {"feat": feats}


class AtomFeaturizerFull(BaseFeaturizer):
    """
    Featurize atoms in a molecule.

    The atom indices will be preserved, i.e. feature i corresponds to atom i.
    """

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
        try:
            species = sorted(kwargs["dataset_species"])
        except KeyError as e:
            raise KeyError(
                "{} `dataset_species` needed for {}.".format(e, self.__class__.__name__)
            )

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

        ring = mol.GetRingInfo()
        allowed_ring_size = [3, 4, 5, 6, 7]
        num_atoms = mol.GetNumAtoms()
        for u in range(num_atoms):
            ft = [is_acceptor[u], is_donor[u]]

            atom = mol.GetAtomWithIdx(u)

            # ft.append(atom.GetDegree())
            ft.append(atom.GetTotalDegree())

            # ft.append(atom.GetExplicitValence())
            # ft.append(atom.GetImplicitValence())
            ft.append(atom.GetTotalValence())

            # ft.append(atom.GetFormalCharge())
            ft.append(atom.GetNumRadicalElectrons())

            ft.append(int(atom.GetIsAromatic()))
            ft.append(int(atom.IsInRing()))

            # ft.append(atom.GetNumExplicitHs())
            # ft.append(atom.GetNumImplicitHs())
            ft.append(atom.GetTotalNumHs(includeNeighbors=True))

            # ft.append(atom.GetAtomicNum())
            ft += one_hot_encoding(atom.GetSymbol(), species)

            ft += one_hot_encoding(
                atom.GetHybridization(),
                [
                    Chem.rdchem.HybridizationType.S,
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    # Chem.rdchem.HybridizationType.SP3D,
                    # Chem.rdchem.HybridizationType.SP3D2,
                ],
            )

            for s in allowed_ring_size:
                ft.append(ring.IsAtomInRingOfSize(u, s))

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
            + ["chemical symbol"] * len(species)
            + ["hybridization"] * 4
            + ["ring size"] * 5
        )

        return {"feat": feats}


class GlobalFeaturizer(BaseFeaturizer):
    """
    Featurize the global state of a molecules using charge only.

    Args:
        allowed_charges (list, optional): charges allowed the the molecules to take.
    """

    def __init__(self, allowed_charges=None, dtype="float32"):
        super(GlobalFeaturizer, self).__init__(dtype)
        self.allowed_charges = allowed_charges

    def __call__(self, mol, **kwargs):

        pt = GetPeriodicTable()
        g = [
            mol.GetNumAtoms(),
            mol.GetNumBonds(),
            sum([pt.GetAtomicWeight(a.GetAtomicNum()) for a in mol.GetAtoms()]),
        ]

        if self.allowed_charges is not None:
            try:
                feats_info = kwargs["extra_feats_info"]
            except KeyError as e:
                raise KeyError(
                    "{} `extra_feats_info` needed for {}.".format(
                        e, self.__class__.__name__
                    )
                )
            g += one_hot_encoding(feats_info["charge"], self.allowed_charges)

        feats = torch.tensor([g], dtype=getattr(torch, self.dtype))

        self._feature_size = feats.shape[1]
        self._feature_name = ["num atoms", "num bonds", "molecule weight"]
        if self.allowed_charges is not None:
            self._feature_name += ["charge one hot"] * len(self.allowed_charges)

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


def multi_hot_encoding(x, allowable_set):
    """Multi-hot encoding.

    Args:
        x (list): any type that can be compared with elements in allowable_set
        allowable_set (list): allowed values for x to take

    Returns:
        list: List of int (0 or 1) where zero or more values can be 1.
            If the i-th value is 1, then we must have allowable_set[i] in x.
    """
    return list(map(int, list(map(lambda s: s in x, allowable_set))))


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
        super(DistanceBins, self).__init__()
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
    e(d) = exp(- gamma * ||d - mu_k||^2), where gamma = 1/delta

    Parameters
    ----------
    low : float
        Smallest value to take for mu_k, default to be 0.
    high : float
        Largest value to take for mu_k, default to be 4.
    num_centers : float
        Number of centers
    """

    def __init__(self, low=0.0, high=4.0, num_centers=20):
        super(RBF, self).__init__()
        self.num_centers = num_centers
        self.centers = np.linspace(low, high, num_centers)
        self.gap = self.centers[1] - self.centers[0]

    @property
    def feature_size(self):
        return self.num_centers

    @property
    def feature_name(self):
        return ["rbf"] * self.feature_size

    def __call__(self, edge_distance):
        """
        Parameters
        ----------
        edge_distance : float
            Edge distance
        Returns
        -------
        a list of RBF values of size `num_centers`
        """
        radial = edge_distance - self.centers
        coef = -1 / self.gap
        return list(np.exp(coef * (radial ** 2)))
