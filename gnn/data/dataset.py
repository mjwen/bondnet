"""
The Li-EC electorlyte dataset.
"""
# pylint: disable=not-callable,no-member

import numpy as np
import torch
import os
import logging
from gnn.utils import expand_path, pickle_dump, pickle_load
from gnn.data.featurizer import (
    AtomFeaturizer,
    BondFeaturizer,
    GlobalStateFeaturizer,
    HeteroMoleculeGraph,
)

try:
    from rdkit import Chem
    from rdkit.Chem import rdmolops
except ImportError:
    pass


logger = logging.getLogger(__name__)


class BaseDataset:
    """
    Base dataset class.
    """

    def __init__(self, dtype="float32"):
        if dtype not in ["float32", "float64"]:
            raise ValueError(
                "`dtype` should be `float32` or `float64`, but got `{}`.".format(dtype)
            )
        self.dtype = dtype
        self.graphs = None
        self.labels = None
        self._feature_size = None

    @property
    def feature_size(self):
        """
        Returns a dict of feature size with node type as the key.
        """
        return self._feature_size

    def get_feature_size(self, ntypes):
        """
        Returns a list of the feature corresponding to the note types `ntypes`.
        """
        size = []
        for n in ntypes:
            for k in self.feature_size:
                if n in k:
                    size.append(self.feature_size[k])
        # TODO more checks needed e.g. one node get more than one size
        msg = "cannot get feature size for nodes: {}".format(ntypes)
        assert len(ntypes) == len(size), msg
        return size

    def __getitem__(self, item):
        """Get datapoint with index

        Args:
            item (int): Datapoint index

        Returns:
            g: DGLHeteroGraph for the ith datapoint
            lb (dict): Labels of the datapoint
        """
        g, lb = self.graphs[item], self.labels[item]
        return g, lb

    def __len__(self):
        """Length of the dataset

        Returns:
            Length of Dataset
        """
        return len(self.graphs)


class ElectrolyteDataset(BaseDataset):
    """
    The electrolyte dataset for Li-ion battery.

    Args:
        sdf_file (str): path to the sdf file of the molecules. Preprocessed dataset
            can be stored in a pickle file (e.g. with file extension of `pkl`) and
            provided for fast recovery.
        label_file (str): path to the label file. Similar to the sdf_file, pickled file
            can be provided for fast recovery.
    """

    def __init__(self, sdf_file, label_file, pickle_dataset=False):
        super(ElectrolyteDataset, self).__init__()
        self.sdf_file = sdf_file
        self.label_file = label_file
        self.pickle_dataset = pickle_dataset

        if self._is_pickled(self.sdf_file) != self._is_pickled(self.label_file):
            raise ValueError("sdf file and label file does not have the same format")
        if self._is_pickled(self.sdf_file):
            self._pickled = True
        else:
            self._pickled = False

        self._feature_size = None

        self._load()

    @property
    def feature_size(self):
        return self._feature_size

    def _load(self):
        logger.info(
            "Start loading dataset from {} and {}...".format(
                self.sdf_file, self.label_file
            )
        )

        if self._pickled:
            self.graphs = pickle_load(self.sdf_file)
            self.labels = pickle_load(self.label_file)
            filename = self._default_stat_dict_filename()
            self._feature_size = self.load_stat_dict(filename)
        else:

            properties = self._read_label_file()
            supp = Chem.SDMolSupplier(self.sdf_file, sanitize=True, removeHs=False)
            species = self._get_species()
            dataset_size = len(properties)

            atom_featurizer = AtomFeaturizer(species, dtype=self.dtype)
            bond_featurizer = BondFeaturizer(dtype=self.dtype)
            global_featiruzer = GlobalStateFeaturizer(dtype=self.dtype)

            self.graphs = []
            self.labels = []
            for i, (mol, prop) in enumerate(zip(supp, properties)):

                if i % 100 == 0:
                    logger.info("Processing molecule {}/{}".format(i, dataset_size))

                mol = rdmolops.AddHs(mol, explicitOnly=True)

                charge = prop[0]
                nbonds = int((len(prop) - 1) / 2)
                dtype = getattr(torch, self.dtype)
                bonds_energy = torch.tensor(prop[1 : nbonds + 1], dtype=dtype)
                bonds_indicator = torch.tensor(prop[nbonds + 1 :], dtype=dtype)

                grapher = HeteroMoleculeGraph(
                    atom_featurizer=atom_featurizer,
                    bond_featurizer=bond_featurizer,
                    global_state_featurizer=global_featiruzer,
                )
                g = grapher.build_graph_and_featurize(mol, charge)
                # TODO the smiles can be removed (if we want to attach some thing, we can
                # attach the moloid)
                smile = Chem.MolToSmiles(mol)
                g.smile = smile
                self.graphs.append(g)

                label = {"energies": bonds_energy, "indicators": bonds_indicator}
                self.labels.append(label)

            self._feature_size = {
                "atom_featurizer": atom_featurizer.feature_size,
                "bond_featurizer": bond_featurizer.feature_size,
                "global_featurizer": global_featiruzer.feature_size,
            }
            if self.pickle_dataset:
                self.save_dataset()
                filename = self._default_stat_dict_filename()
                self.save_stat_dict(filename)

        logger.info("Finish loading {} graphs...".format(len(self.labels)))

    def save_dataset(self):
        filename = os.path.splitext(self.sdf_file)[0] + ".pkl"
        pickle_dump(self.graphs, filename)
        filename = os.path.splitext(self.label_file)[0] + ".pkl"
        pickle_dump(self.labels, filename)

    def _default_stat_dict_filename(self):
        filename = expand_path(self.sdf_file)
        return os.path.join(
            os.path.dirname(filename), self.__class__.__name__ + "_stat_dict.pkl"
        )

    def load_stat_dict(self, filename):
        return pickle_load(filename)

    def save_stat_dict(self, filename):
        pickle_dump(self._feature_size, filename)

    def _read_label_file(self):
        rslt = []
        with open(self.label_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue
                line = [float(i) for i in line.split()]
                rslt.append(line)
        return rslt

    def _get_species(self):
        suppl = Chem.SDMolSupplier(self.sdf_file, sanitize=True, removeHs=False)
        system_species = set()
        for i, mol in enumerate(suppl):
            try:
                atoms = mol.GetAtoms()
                species = [a.GetSymbol() for a in atoms]
                system_species.update(species)
            except AttributeError:
                raise RuntimeError("Error reading mol '{}' from sdf file.".format(i))

        return list(system_species)

    # TODO we may need to implement normalization in featurizer and provide a wrapper
    # here. But it seems we do not need to normalize label
    # def set_mean_and_std(self, mean=None, std=None):
    #     """Set mean and std or compute from labels for future normalization.

    #     Parameters
    #     ----------
    #     mean : int or float
    #         Default to be None.
    #     std : int or float
    #         Default to be None.
    #     """
    #     labels = np.array([i.numpy() for i in self.labels])
    #     if mean is None:
    #         mean = np.mean(labels, axis=0)
    #     if std is None:
    #         std = np.std(labels, axis=0)
    #     self.mean = mean
    #     self.std = std

    @staticmethod
    def _is_pickled(filename):
        filename = expand_path(filename)
        if os.path.splitext(filename)[1] == "pkl":
            return True
        else:
            return False


class Subset(BaseDataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self._feature_size = dataset.feature_size

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def train_validation_test_split(dataset, validation=0.1, test=0.1, random_seed=35):
    """
    Split a dataset into training, validation, and test set.

    The training set will be automatically determined based on `validation` and `test`,
    i.e. train = 1 - validation - test.

    Args:
        dataset: the dataset
        validation (float, optional): The amount of data (fraction) to be assigned to
            validation set. Defaults to 0.1.
        test (float, optional): The amount of data (fraction) to be assigned to test
            set. Defaults to 0.1.
        random_seed (int, optional): random seed that determines the permutation of the
            dataset. Defaults to 35.

    Returns:
        [train set, validation set, test_set]
    """
    assert validation + test < 1.0, "valication + test >= 1"
    size = len(dataset)
    num_val = int(size * validation)
    num_test = int(size * test)
    num_train = size - num_val - num_test

    np.random.seed(random_seed)
    idx = np.random.permutation(size)
    train_idx = idx[:num_train]
    val_idx = idx[num_train : num_train + num_val]
    test_idx = idx[num_train + num_val :]
    return [
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    ]
