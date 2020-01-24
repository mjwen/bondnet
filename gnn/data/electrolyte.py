"""
The Li-EC electrolyte dataset.
"""

import torch
import os
import logging
from rdkit import Chem
from gnn.utils import expand_path, pickle_dump, pickle_load
from gnn.data.featurizer import (
    AtomFeaturizer,
    AtomFeaturizerWithExtraInfo,
    BondAsNodeFeaturizer,
    BondAsEdgeBidirectedFeaturizer,
    BondAsEdgeCompleteFeaturizer,
    GlobalFeaturizerWithExtraInfo,
)
from gnn.data.grapher import HomoBidirectedGraph, HomoCompleteGraph, HeteroMoleculeGraph
from gnn.data.dataset import BaseDataset
from gnn.data.transformers import StandardScaler, GraphFeatureStandardScaler
from gnn.utils import yaml_load


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
        feature_file (str): path to the feature file. If `None` features will be
            calculated only using rdkit. Otherwise, features can be provided through this
            file.
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
        feature_file=None,
        self_loop=True,
        grapher="hetero",
        bond_length_featurizer=None,
        feature_transformer=True,
        label_transformer=True,
        pickle_dataset=False,
        dtype="float32",
    ):
        super(ElectrolyteDataset, self).__init__(dtype)
        self.sdf_file = expand_path(sdf_file)
        self.label_file = expand_path(label_file)
        if feature_file is None:
            self.feature_file = None
        else:
            self.feature_file = expand_path(feature_file)

        self.self_loop = self_loop
        self.grapher = grapher
        self.bond_length_featurizer = bond_length_featurizer
        self.feature_transformer = feature_transformer
        self.label_transformer = label_transformer
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

            if self.grapher == "hetero":
                self.graphs, self.labels = self.load_dataset_hetero()
            elif self.grapher in ["homo_bidirected", "homo_complete"]:
                self.graphs, self.labels = self.load_dataset()
            else:
                raise ValueError("Unsupported grapher type '{}".format(self.grapher))

            self.load_state_dict(self._default_state_dict_filename())

        else:
            logger.info(
                "Start loading dataset from files {} and {}...".format(
                    self.sdf_file, self.label_file
                )
            )

            # initialize featurizer
            species = self._get_species()

            if self.feature_file is not None:
                atom_featurizer = AtomFeaturizerWithExtraInfo(species, dtype=self.dtype)
            else:
                atom_featurizer = AtomFeaturizer(species, dtype=self.dtype)
            if self.grapher == "hetero":
                bond_featurizer = BondAsNodeFeaturizer(
                    length_featurizer=self.bond_length_featurizer, dtype=self.dtype
                )
                global_featurizer = GlobalFeaturizerWithExtraInfo(dtype=self.dtype)
                grapher = HeteroMoleculeGraph(
                    atom_featurizer=atom_featurizer,
                    bond_featurizer=bond_featurizer,
                    global_state_featurizer=global_featurizer,
                    self_loop=self.self_loop,
                )
            elif self.grapher == "homo_bidirected":
                bond_featurizer = BondAsEdgeBidirectedFeaturizer(
                    self_loop=self.self_loop,
                    length_featurizer=self.bond_length_featurizer,
                    dtype=self.dtype,
                )
                grapher = HomoBidirectedGraph(
                    atom_featurizer=atom_featurizer,
                    bond_featurizer=bond_featurizer,
                    self_loop=self.self_loop,
                )

            elif self.grapher == "homo_complete":
                bond_featurizer = BondAsEdgeCompleteFeaturizer(
                    self_loop=self.self_loop,
                    length_featurizer=self.bond_length_featurizer,
                    dtype=self.dtype,
                )
                grapher = HomoCompleteGraph(
                    atom_featurizer=atom_featurizer,
                    bond_featurizer=bond_featurizer,
                    self_loop=self.self_loop,
                )
            else:
                raise ValueError("Unsupported grapher type '{}".format(self.grapher))

            # read graphs and label
            properties = self._read_label_file()

            # additional features from file
            if self.feature_file is not None:
                features = self._read_feature_file()
            else:
                features = [None] * len(properties)

            self.graphs = []
            self.labels = []
            supp = Chem.SDMolSupplier(self.sdf_file, sanitize=True, removeHs=False)

            for i, (mol, feats, prop) in enumerate(zip(supp, features, properties)):
                if i % 100 == 0:
                    logger.info("Processing molecule {}/{}".format(i, len(properties)))

                if mol is None:  # bad mol
                    continue

                nbonds = int((len(prop) - 1) / 2)
                dtype = getattr(torch, self.dtype)
                bonds_energy = torch.tensor(prop[1 : nbonds + 1], dtype=dtype)
                bonds_indicator = torch.tensor(prop[nbonds + 1 :], dtype=dtype)

                if feats is not None:
                    g = grapher.build_graph_and_featurize(mol, extra_feats_info=feats)
                else:
                    g = grapher.build_graph_and_featurize(mol)

                self.graphs.append(g)

                label = {"value": bonds_energy, "indicator": bonds_indicator}
                self.labels.append(label)

            # transformers
            if self.feature_transformer:
                feature_scaler = GraphFeatureStandardScaler()
                self.graphs = feature_scaler(self.graphs)
                logger.info("Feature scaler mean: {}".format(feature_scaler.mean))
                logger.info("Feature scaler std: {}".format(feature_scaler.std))

            # labels are standardized by y' = (y - mean(y))/std(y), the model will be
            # trained on this scaled value. However for metric measure (e.g. MAE) we need
            # to convert y' back to y, i.e. y = y' * std(y) + mean(y), the model
            # predition is then y^ = y'^ *std(y) + mean(y), where ^ means predictions.
            # Then MAE is |y^-y| = |y'^ - y'| *std(y), i.e. we just need to multiple
            # standard deviation to get back to the original scale. Similar analysis
            # applies to RMSE.
            if self.label_transformer:
                labels = [lb["value"] for lb in self.labels]  # list of 1D tensor
                labels = torch.cat(labels)  # 1D tensor
                labels = labels.view(len(labels), 1)  # 2D tensor of shape (N, 1)

                label_scaler = StandardScaler()
                labels = label_scaler(labels)
                labels = torch.tensor(labels, dtype=getattr(torch, self.dtype))

                sizes = [len(lb["value"]) for lb in self.labels]
                labels = torch.split(labels, sizes)  # list of 2D tensor of shape (Nb, 1)
                labels = [torch.flatten(lb) for lb in labels]

                std = label_scaler.std[0]
                self.transformer_scale = []
                for i, lb in enumerate(labels):
                    self.labels[i]["value"] = lb
                    sca = torch.tensor([std] * len(lb), dtype=getattr(torch, self.dtype))
                    self.transformer_scale.append(sca)

                logger.info("Label scaler mean: {}".format(label_scaler.mean))
                logger.info("Label scaler std: {}".format(label_scaler.std))

            self._feature_size = {
                "atom": atom_featurizer.feature_size,
                "bond": bond_featurizer.feature_size,
            }
            self._feature_name = {
                "atom": atom_featurizer.feature_name,
                "bond": bond_featurizer.feature_name,
            }
            if self.grapher == "hetero":
                self._feature_size["global"] = global_featurizer.feature_size
                self._feature_name["global"] = global_featurizer.feature_name
            logger.info("Feature size: {}".format(self._feature_size))
            logger.info("Feature name: {}".format(self._feature_name))

            if self.pickle_dataset:
                if self.grapher == "hetero":
                    self.save_dataset_hetero()
                else:
                    self.save_dataset()
                self.save_state_dict(self._default_state_dict_filename())

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

    def load_state_dict(self, filename):
        d = pickle_load(filename)
        self._feature_size = d["feature_size"]
        self._feature_name = d["feature_name"]
        self.transformer_scale = d["transformer_scale"]

    def save_state_dict(self, filename):
        d = {
            "feature_size": self._feature_size,
            "feature_name": self._feature_name,
            "transformer_scale": self.transformer_scale,
        }
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

                # remove inline comments
                if "#" in line:
                    line = line[: line.index("#")]

                line = [float(i) for i in line.split()]
                rslt.append(line)
        return rslt

    def _read_feature_file(self):
        return yaml_load(self.feature_file)

    def __repr__(self):
        rst = "Dataset " + self.__class__.__name__ + "\n"
        rst += "Length: {}\n".format(len(self))
        for ft, sz in self.feature_size.items():
            rst += "Feature: {}, size: {}\n".format(ft, sz)
        for ft, nm in self.feature_name.items():
            rst += "Feature: {}, name: {}\n".format(ft, nm)
        return rst
