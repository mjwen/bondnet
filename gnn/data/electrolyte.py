"""
The Li-EC electrolyte dataset.
"""
# pylint: disable=not-callable,no-member

import torch
import os
import logging
from rdkit import Chem
from gnn.utils import expand_path, pickle_dump, pickle_load
from gnn.data.featurizer import (
    AtomFeaturizer,
    BondAsNodeFeaturizer,
    BondAsEdgeBidirectedFeaturizer,
    MolChargeFeaturizer,
)
from gnn.data.grapher import HomoBidirectedGraph, HeteroMoleculeGraph
from gnn.data.dataset import BaseDataset


logger = logging.getLogger(__name__)


class ElectrolyteDataset(BaseDataset):
    """
    The electrolyte dataset for Li-ion battery.

    Args:
        sdf_file (str): path to the sdf file of the molecules. Preprocessed dataset
            can be stored in a pickle file (e.g. with file extension of `pkl`) and
            provided for fast recovery.
        label_file (str): path to the label file. Similar to the sdf_file, pickled file
            can be provided for fast recovery.
        self_loop (bool): whether to create self loop, i.e. a node is connected to
            itself through an edge.
        hetero (bool): Whether to build hetero graph, where atom, bond, and global state
            are all represented as graph nodes). If `False`, build a homo graph, where
            atoms are represented as node and bond are represented as graphs.
    """

    def __init__(
        self,
        sdf_file,
        label_file,
        self_loop=True,
        hetero=True,
        pickle_dataset=False,
        dtype="float32",
    ):
        super(ElectrolyteDataset, self).__init__(dtype)
        self.sdf_file = expand_path(sdf_file)
        self.label_file = expand_path(label_file)
        self.self_loop = self_loop
        self.hetero = hetero
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
        if self._pickled:
            logger.info(
                "Start loading dataset from picked files {} and {}...".format(
                    self.sdf_file, self.label_file
                )
            )

            if self.hetero:
                self.graphs, self.labels = self.load_dataset_hetero()
            else:
                self.graphs, self.labels = self.load_dataset()
            d = self.load_state_dict(self._default_state_dict_filename())
            self._feature_size = d["feature_size"]
            self._feature_name = d["feature_name"]

        else:
            logger.info(
                "Start loading dataset from files {} and {}...".format(
                    self.sdf_file, self.label_file
                )
            )

            species = self._get_species()
            atom_featurizer = AtomFeaturizer(species, dtype=self.dtype)
            if self.hetero:
                global_featurizer = MolChargeFeaturizer(dtype=self.dtype)
                bond_featurizer = BondAsNodeFeaturizer(dtype=self.dtype)
                grapher = HeteroMoleculeGraph(
                    atom_featurizer=atom_featurizer,
                    bond_featurizer=bond_featurizer,
                    global_state_featurizer=global_featurizer,
                    self_loop=self.self_loop,
                )
            else:
                bond_featurizer = BondAsEdgeBidirectedFeaturizer(dtype=self.dtype)
                grapher = HomoBidirectedGraph(
                    atom_featurizer=atom_featurizer,
                    bond_featurizer=bond_featurizer,
                    self_loop=self.self_loop,
                )

            properties = self._read_label_file()

            self.graphs = []
            self.labels = []
            supp = Chem.SDMolSupplier(self.sdf_file, sanitize=True, removeHs=False)

            for i, (mol, prop) in enumerate(zip(supp, properties)):
                if i % 100 == 0:
                    logger.info("Processing molecule {}/{}".format(i, len(properties)))

                if mol is None:  # bad mol
                    continue

                nbonds = int((len(prop) - 1) / 2)
                dtype = getattr(torch, self.dtype)
                bonds_energy = torch.tensor(prop[1 : nbonds + 1], dtype=dtype)
                bonds_indicator = torch.tensor(prop[nbonds + 1 :], dtype=dtype)

                if self.hetero:
                    g = grapher.build_graph_and_featurize(mol, charge=prop[0])
                else:
                    g = grapher.build_graph_and_featurize(mol)
                self.graphs.append(g)

                label = {"value": bonds_energy, "indicator": bonds_indicator}
                self.labels.append(label)

            self._feature_size = {
                "atom": atom_featurizer.feature_size,
                "bond": bond_featurizer.feature_size,
            }
            self._feature_name = {
                "atom": atom_featurizer.feature_name,
                "bond": bond_featurizer.feature_name,
            }
            if self.hetero:
                self._feature_size["global"] = global_featurizer.feature_size
                self._feature_name["global"] = global_featurizer.feature_name

            if self.pickle_dataset:
                if self.hetero:
                    self.save_dataset_hetero()
                else:
                    self.save_dataset()
                filename = self._default_state_dict_filename()
                self.save_state_dict(filename)

        logger.info("Finish loading {} graphs...".format(len(self.labels)))

    def save_dataset(self):
        filename = self.sdf_file + ".pkl"
        pickle_dump(self.graphs, filename)
        filename = self.label_file + ".pkl"
        pickle_dump(self.labels, filename)

    def load_dataset(self):
        graphs = pickle_load(self.sdf_file)
        labels = pickle_load(self.label_file)
        return graphs, labels

    # NOTE currently, DGLHeterograph does not support pickle, so we pickle the data only
    # and we can get back to the above two functions once it is supported
    def save_dataset_hetero(self):
        filename = self.sdf_file + ".pkl"
        data = []
        for g in self.graphs:
            ndata = {t: dict(g.nodes[t].data) for t in g.ntypes}
            edata = {t: dict(g.edges[t].data) for t in g.etypes}
            data.append([ndata, edata])
        pickle_dump(data, filename)
        filename = self.label_file + ".pkl"
        pickle_dump(self.labels, filename)

    def load_dataset_hetero(self):
        data = pickle_load(self.sdf_file)
        fname = self.sdf_file.replace(".pkl", "")
        supp = Chem.SDMolSupplier(fname, sanitize=True, removeHs=False)

        graphs = []
        i = 0
        for mol in supp:
            if mol is None:  # bad mol
                continue
            entry = data[i]
            i += 1

            grapher = HeteroMoleculeGraph(self_loop=self.self_loop)
            g = grapher.build_graph(mol)
            for t, v in entry[0].items():
                g.nodes[t].data.update(v)
            for t, v in entry[1].items():
                g.edges[t].data.update(v)
            graphs.append(g)

        labels = pickle_load(self.label_file)

        return graphs, labels

    def _default_state_dict_filename(self):
        filename = expand_path(self.sdf_file)
        return os.path.join(
            os.path.dirname(filename), self.__class__.__name__ + "_state_dict.pkl"
        )

    @staticmethod
    def load_state_dict(filename):
        return pickle_load(filename)

    def save_state_dict(self, filename):
        d = {"feature_size": self._feature_size, "feature_name": self._feature_name}
        pickle_dump(d, filename)

    def _get_species(self):
        suppl = Chem.SDMolSupplier(self.sdf_file, sanitize=True, removeHs=False)
        system_species = set()
        for i, mol in enumerate(suppl):
            if mol is None:
                logger.info("bad mol {}.".format(i))
                continue
            atoms = mol.GetAtoms()
            species = [a.GetSymbol() for a in atoms]
            system_species.update(species)
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
        if os.path.splitext(filename)[1] == ".pkl":
            return True
        else:
            return False

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
