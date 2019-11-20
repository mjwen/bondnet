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


class ElectrolyteDataset:
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

    def get_feature_size(self, ntypes):
        size = []
        for n in ntypes:
            for k in self.feature_size:
                if n in k:
                    size.append(self.feature_size[k])
        # TODO more checks needed e.g. one node get more than one size
        msg = "cannot get feature size for nodes: {}".format(ntypes)
        assert len(ntypes) == len(size), msg
        return size

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

            atom_featurizer = AtomFeaturizer(species)
            bond_featurizer = BondFeaturizer()
            global_featiruzer = GlobalStateFeaturizer()

            self.graphs = []
            self.labels = []
            for i, (mol, prop) in enumerate(zip(supp, properties)):

                if i % 100 == 0:
                    logger.info("Processing molecule {}/{}".format(i, dataset_size))

                mol = rdmolops.AddHs(mol, explicitOnly=True)

                charge = prop[0]
                nbonds = int((len(prop) - 1) / 2)
                bonds_energy = torch.from_numpy(
                    np.asarray(prop[1 : nbonds + 1], dtype=np.float32)
                )
                bonds_indicator = torch.from_numpy(
                    np.asarray(prop[nbonds + 1 :], dtype=np.float32)
                )

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

    def __getitem__(self, item):
        """Get datapoint with index

        Args:
            item (int): Datapoint index

        Returns:
            g: DGLHeteroGraph for the ith datapoint
            l (dict): Labels of the datapoint for all tasks
        """
        g, la = self.graphs[item], self.labels[item]
        return g, la

    def __len__(self):
        """Length of the dataset

        Returns:
            Length of Dataset
        """
        return len(self.graphs)

