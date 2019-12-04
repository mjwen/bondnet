"""
The Li-EC electrolyte dataset.
"""
# pylint: disable=not-callable,no-member

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
from gnn.data.dataset import BaseDataset
import pandas as pd
import numpy as np
import warnings

try:
    from rdkit import Chem
    from rdkit.Chem import rdmolops
except ImportError:
    pass


logger = logging.getLogger(__name__)


class QM9Dataset(BaseDataset):
    """
    The QM9 dataset.

    Args:
        sdf_file (str): path to the sdf file of the molecules. Preprocessed dataset
            can be stored in a pickle file (e.g. with file extension of `pkl`) and
            provided for fast recovery.
        label_file (str): path to the label file. Similar to the sdf_file, pickled file
            can be provided for fast recovery.
        self_loop (bool): whether to create self loop, i.e. a node is connected to
            itself through an edge.
    """

    def __init__(
        self, sdf_file, label_file, self_loop=True, pickle_dataset=False, dtype="float32"
    ):
        super(QM9Dataset, self).__init__(dtype)
        self.sdf_file = sdf_file
        self.label_file = label_file
        self.self_loop = self_loop
        self.pickle_dataset = pickle_dataset

        if self._is_pickled(self.sdf_file) != self._is_pickled(self.label_file):
            raise ValueError("sdf file and label file does not have the same format")
        if self._is_pickled(self.sdf_file):
            self._pickled = True
        else:
            self._pickled = False

        self._load()

    @property
    def feature_size(self):
        return self._feature_size

    @property
    def feature_name(self):
        return self._feature_name

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
            d = self.load_stat_dict(filename)
            self._feature_size = d["feature_size"]
            self._feature_name = d["feature_name"]
        else:
            species, bad_mols = self._get_species()
            atom_featurizer = AtomFeaturizer(species, dtype=self.dtype)
            bond_featurizer = BondFeaturizer(dtype=self.dtype)
            global_featurizer = GlobalStateFeaturizer(dtype=self.dtype)

            labels = np.delete(self._read_label_file(), bad_mols, axis=0)
            self.labels = torch.tensor(labels, dtype=getattr(torch, self.dtype))
            dataset_size = len(self.labels)

            self.graphs = []
            supp = Chem.SDMolSupplier(self.sdf_file, sanitize=False, removeHs=False)
            for i, mol in enumerate(supp):
                if i % 100 == 0:
                    logger.info("Processing molecule {}/{}".format(i, dataset_size))

                mol = rdmolops.AddHs(mol, explicitOnly=True)
                if mol is None:  # bad mol
                    continue

                grapher = HeteroMoleculeGraph(
                    atom_featurizer=atom_featurizer,
                    bond_featurizer=bond_featurizer,
                    global_state_featurizer=global_featurizer,
                    self_loop=self.self_loop,
                )
                g = grapher.build_graph_and_featurize(mol, charge=0)
                self.graphs.append(g)

            self._feature_size = {
                "atom": atom_featurizer.feature_size,
                "bond": bond_featurizer.feature_size,
                "global": global_featurizer.feature_size,
            }
            self._feature_name = {
                "atom": atom_featurizer.feature_name,
                "bond": bond_featurizer.feature_name,
                "global": global_featurizer.feature_name,
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
        d = {"feature_size": self._feature_size, "feature_name": self._feature_name}
        pickle_dump(d, filename)

    def _read_label_file(self):
        rst = pd.read_csv(self.label_file, index_col=0)
        rst = rst.to_numpy()
        return rst

    def _get_species(self):
        suppl = Chem.SDMolSupplier(self.sdf_file, sanitize=False, removeHs=False)
        system_species = set()
        bad_mols = []
        for i, mol in enumerate(suppl):
            try:
                atoms = mol.GetAtoms()
                species = [a.GetSymbol() for a in atoms]
                system_species.update(species)
            except AttributeError:
                bad_mols.append(i)
                warnings.warn("Error reading mol '{}' from sdf file.".format(i))

        return list(system_species), bad_mols

    @staticmethod
    def _is_pickled(filename):
        filename = expand_path(filename)
        if os.path.splitext(filename)[1] == "pkl":
            return True
        else:
            return False
