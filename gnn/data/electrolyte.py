"""
The Li-EC electrolyte dataset.
"""

import torch
import logging
import numpy as np
from collections import OrderedDict
import pandas as pd
from rdkit import Chem
from gnn.data.dataset import BaseDataset
from gnn.data.transformers import StandardScaler, GraphFeatureStandardScaler
from gnn.utils import expand_path, yaml_load


logger = logging.getLogger(__name__)


class ElectrolyteBondDataset(BaseDataset):
    """
    The electrolyte dataset for Li-ion battery.

    Args:
        grapher (object): grapher object that build different types of graphs:
            `hetero`, `homo_bidirected` and `homo_complete`.
            For hetero graph, atom, bond, and global state are all represented as
            graph nodes. For homo graph, atoms are represented as node and bond are
            represented as graph edges.
        sdf_file (str): path to the sdf file of the molecules. Preprocessed dataset
            can be stored in a pickle file (e.g. with file extension of `pkl`) and
            provided for fast recovery.
        label_file (str): path to the label file. Similar to the sdf_file, pickled file
            can be provided for fast recovery.
        feature_file (str): path to the feature file. If `None` features will be
            calculated only using rdkit. Otherwise, features can be provided through this
            file.
    """

    def __init__(
        self,
        grapher,
        sdf_file,
        label_file,
        feature_file=None,
        feature_transformer=True,
        label_transformer=True,
        pickle_dataset=False,
        dtype="float32",
    ):
        super(ElectrolyteBondDataset, self).__init__(dtype)
        self.grapher = grapher
        self.sdf_file = expand_path(sdf_file)
        self.label_file = expand_path(label_file)
        self.feature_file = None if feature_file is None else expand_path(feature_file)
        self.feature_transformer = feature_transformer
        self.label_transformer = label_transformer
        self.pickle_dataset = pickle_dataset

        ### pickle related
        # if self._is_pickled(self.sdf_file) != self._is_pickled(self.label_file):
        #     raise ValueError("sdf file and label file does not have the same format")
        # if self._is_pickled(self.sdf_file):
        #     self._pickled = True
        # else:
        #     self._pickled = False

        self._load()

    @property
    def feature_size(self):
        return self._feature_size

    @property
    def feature_name(self):
        return self._feature_name

    def _load(self):
        ### pickle related
        # if self._pickled:
        #     logger.info(
        #         "Start loading dataset from picked files {} and {}...".format(
        #             self.sdf_file, self.label_file
        #         )
        #     )
        #
        #     if self.grapher == "hetero":
        #         self.graphs, self.labels = self.load_dataset_hetero()
        #     elif self.grapher in ["homo_bidirected", "homo_complete"]:
        #         self.graphs, self.labels = self.load_dataset()
        #     else:
        #         raise ValueError("Unsupported grapher type '{}".format(self.grapher))
        #
        #     self.load_state_dict(self._default_state_dict_filename())
        #
        #     return
        #

        logger.info(
            "Start loading dataset from files {}, {}...".format(
                self.sdf_file, self.label_file
            )
        )

        # get species of dataset
        species = self._get_species()

        # read graphs and label
        raw_value, raw_indicator, raw_mol_source = self._read_label_file()

        # additional features from file
        if self.feature_file is not None:
            features = self._read_feature_file()
        else:
            features = [None] * len(raw_value)

        self.graphs = []
        self.labels = []
        supp = Chem.SDMolSupplier(self.sdf_file, sanitize=True, removeHs=False)

        for i, mol in enumerate(supp):
            if i % 100 == 0:
                logger.info("Processing molecule {}/{}".format(i, len(raw_value)))

            # bad mol
            if mol is None:
                continue

            # graph
            g = self.grapher.build_graph_and_featurize(
                mol, extra_feats_info=features[i], dataset_species=species
            )
            # we add this for check purpose, because some entries in the sdf file may fail
            g.graph_id = i
            self.graphs.append(g)

            # label
            dtype = getattr(torch, self.dtype)
            bonds_energy = torch.tensor(raw_value[i], dtype=dtype)
            bonds_indicator = torch.tensor(raw_indicator[i], dtype=dtype)
            bonds_mol_source = raw_mol_source[i]
            label = {
                "value": bonds_energy,
                "indicator": bonds_indicator,
                "mol_source": bonds_mol_source,
            }
            self.labels.append(label)

        # this should be called after grapher.build_graph_and_featurize,
        # which initializes the feature name and size
        self._feature_name = self.grapher.feature_name
        self._feature_size = self.grapher.feature_size
        logger.info("Feature name: {}".format(self.feature_name))
        logger.info("Feature size: {}".format(self.feature_size))

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

            # compute mean and stdev using nonzero elements (these are the actual label)
            non_zeros = [i for i in labels if i != 0.0]
            mean = float(np.mean(non_zeros))
            std = float(np.std(non_zeros))

            # normalization
            labels = (labels - mean) / std
            sizes = [len(lb["value"]) for lb in self.labels]
            labels = torch.split(labels, sizes)

            self.transformer_scale = []
            for i, lb in enumerate(labels):
                self.labels[i]["value"] = lb
                sca = torch.tensor([std] * len(lb), dtype=getattr(torch, self.dtype))
                self.transformer_scale.append(sca)

            logger.info("Label scaler mean: {}".format(mean))
            logger.info("Label scaler std: {}".format(std))

        logger.info("Finish loading {} graphs...".format(len(self.labels)))

        ### pickle related
        # if self.pickle_dataset:
        #     if self.grapher == "hetero":
        #         self.save_dataset_hetero()
        #     else:
        #         self.save_dataset()
        #     self.save_state_dict(self._default_state_dict_filename())

    ### pickle related
    # def save_dataset(self):
    #     filename = self.sdf_file + ".pkl"
    #     pickle_dump(self.graphs, filename)
    #     filename = self.label_file + ".pkl"
    #     pickle_dump(self.labels, filename)
    #
    # def load_dataset(self):
    #     graphs = pickle_load(self.sdf_file)
    #     labels = pickle_load(self.label_file)
    #     return graphs, labels
    #
    # # NOTE currently, DGLHeterograph does not support pickle, so we pickle the data only
    # # and we can get back to the above two functions once it is supported
    # def save_dataset_hetero(self):
    #     filename = self.sdf_file + ".pkl"
    #     data = []
    #     for g in self.graphs:
    #         ndata = {t: dict(g.nodes[t].data) for t in g.ntypes}
    #         edata = {t: dict(g.edges[t].data) for t in g.etypes}
    #         data.append([ndata, edata])
    #     pickle_dump(data, filename)
    #     filename = self.label_file + ".pkl"
    #     pickle_dump(self.labels, filename)
    #
    # def load_dataset_hetero(self):
    #     data = pickle_load(self.sdf_file)
    #     fname = self.sdf_file.replace(".pkl", "")
    #     supp = Chem.SDMolSupplier(fname, sanitize=True, removeHs=False)
    #
    #     graphs = []
    #     i = 0
    #     for mol in supp:
    #         if mol is None:  # bad mol
    #             continue
    #         entry = data[i]
    #         i += 1
    #
    #         grapher = HeteroMoleculeGraph(self_loop=self.self_loop)
    #         g = grapher.build_graph(mol)
    #         for t, v in entry[0].items():
    #             g.nodes[t].data.update(v)
    #         for t, v in entry[1].items():
    #             g.edges[t].data.update(v)
    #         graphs.append(g)
    #
    #     labels = pickle_load(self.label_file)
    #
    #     return graphs, labels
    #
    # def _default_state_dict_filename(self):
    #     filename = expand_path(self.sdf_file)
    #     return os.path.join(
    #         os.path.dirname(filename), self.__class__.__name__ + "_state_dict.pkl"
    #     )
    #
    # def load_state_dict(self, filename):
    #     d = pickle_load(filename)
    #     self._feature_size = d["feature_size"]
    #     self._feature_name = d["feature_file"]
    #     self.transformer_scale = d["transformer_scale"]
    #
    # def save_state_dict(self, filename):
    #     d = {
    #         "feature_size": self._feature_size,
    #         "feature_file": self._feature_name,
    #         "transformer_scale": self.transformer_scale,
    #     }
    #     pickle_dump(d, filename)
    #
    # @staticmethod
    # def _is_pickled(filename):
    #     filename = expand_path(filename)
    #     if os.path.splitext(filename)[1] == ".pkl":
    #         return True
    #     else:
    #         return False

    def _get_species(self):
        suppl = Chem.SDMolSupplier(self.sdf_file, sanitize=True, removeHs=False)
        system_species = set()
        for i, mol in enumerate(suppl):
            if mol is None:
                continue
            atoms = mol.GetAtoms()
            species = [a.GetSymbol() for a in atoms]
            system_species.update(species)
        return list(system_species)

    def _read_label_file(self):
        value = []
        indicator = []
        mol_source = []
        with open(self.label_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue

                # remove inline comments
                if "#" in line:
                    line = line[: line.index("#")]

                line = line.split()
                mol_source.append(line[-1])  # it could be a string

                line = [float(i) for i in line[:-1]]
                nbonds = len(line) // 2
                value.append(line[:nbonds])
                indicator.append(line[nbonds:])

        return value, indicator, mol_source

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


class ElectrolyteMoleculeDataset(ElectrolyteBondDataset):
    def __init__(
        self,
        grapher,
        sdf_file,
        label_file,
        feature_file=None,
        feature_transformer=True,
        label_transformer=True,
        properties=["atomization_energy"],
        unit_conversion=True,
        pickle_dataset=False,
        dtype="float32",
    ):
        self.properties = properties
        self.unit_conversion = unit_conversion
        super(ElectrolyteMoleculeDataset, self).__init__(
            grapher=grapher,
            sdf_file=sdf_file,
            label_file=label_file,
            feature_file=feature_file,
            feature_transformer=feature_transformer,
            label_transformer=label_transformer,
            pickle_dataset=pickle_dataset,
            dtype=dtype,
        )

    def _load(self):

        logger.info(
            "Start loading dataset from files {}, {}...".format(
                self.sdf_file, self.label_file
            )
        )

        # get species of dataset
        species = self._get_species()

        # read mol graphs and label
        raw_labels, extensive = self._read_label_file()

        # additional features from file
        if self.feature_file is not None:
            features = self._read_feature_file()
        else:
            features = [None] * len(raw_labels)

        self.graphs = []
        labels = []
        natoms = []
        supp = Chem.SDMolSupplier(self.sdf_file, sanitize=True, removeHs=False)

        for i, (mol, feats, lb) in enumerate(zip(supp, features, raw_labels)):

            if i % 100 == 0:
                logger.info("Processing molecule {}/{}".format(i, len(raw_labels)))

            if mol is None:
                continue

            # graph
            g = self.grapher.build_graph_and_featurize(
                mol, extra_feats_info=feats, dataset_species=species
            )
            # we add this for check purpose, because some entries in the sdf file may fail
            g.graph_id = i
            self.graphs.append(g)

            # label
            labels.append(lb)
            natoms.append(mol.GetNumAtoms())

        # this should be called after grapher.build_graph_and_featurize,
        # which initializes the feature name and size
        self._feature_name = self.grapher.feature_name
        self._feature_size = self.grapher.feature_size
        logger.info("Feature name: {}".format(self.feature_name))
        logger.info("Feature size: {}".format(self.feature_size))

        # feature and label transformer
        if self.feature_transformer:
            feature_scaler = GraphFeatureStandardScaler()
            self.graphs = feature_scaler(self.graphs)
            logger.info("Feature scaler mean: {}".format(feature_scaler.mean))
            logger.info("Feature scaler std: {}".format(feature_scaler.std))

        if self.label_transformer:
            labels = np.asarray(labels)
            natoms = np.asarray(natoms, dtype=np.float32)

            scaled_labels = []
            transformer_scale = []
            label_scaler_mean = []
            label_scaler_std = []
            for i, is_ext in enumerate(extensive):
                if is_ext:
                    # extensive labels standardized by the number of atoms in the
                    # molecules, i.e. y' = y/natoms
                    lb = labels[:, i] / natoms
                    ts = natoms
                    label_scaler_mean.append(None)
                    label_scaler_std.append("num atoms")
                else:
                    # intensive labels standardized by y' = (y - mean(y))/std(y)
                    scaler = StandardScaler()
                    lb = labels[:, [i]]  # 2D array of shape (N, 1)
                    lb = scaler(lb)
                    lb = lb.ravel()
                    ts = np.repeat(scaler.std, len(lb))
                    label_scaler_mean.append(scaler.mean)
                    label_scaler_std.append(scaler.std)
                scaled_labels.append(lb)
                transformer_scale.append(ts)
            labels = np.asarray(scaled_labels).T

            self.transformer_scale = torch.tensor(
                np.asarray(transformer_scale).T, dtype=getattr(torch, self.dtype)
            )
            logger.info("Label scaler mean: {}".format(label_scaler_mean))
            logger.info("Label scaler std: {}".format(label_scaler_std))

        self.labels = torch.tensor(labels, dtype=getattr(torch, self.dtype))

        logger.info("Finish loading {} graphs...".format(len(self.labels)))

    def _read_label_file(self):
        """
        Returns:
            rst (2D array): shape (N, M), where N is the number of lines (excluding the
                header line), and M is the number of columns (exluding the first index
                column).
            extensive (list): size (M), indicating whether the corresponding data in
                rst is extensive property or not.
        """

        rst = pd.read_csv(self.label_file, index_col=0)
        rst = rst.to_numpy()

        # supported property
        supp_prop = OrderedDict()
        supp_prop["atomization_energy"] = {"uc": 1.0, "extensive": True}

        if self.properties is not None:

            for prop in self.properties:
                if prop not in supp_prop:
                    raise ValueError(
                        "Property '{}' not supported. Supported ones are: {}".format(
                            prop, supp_prop.keys()
                        )
                    )
            supp_list = list(supp_prop.keys())
            indices = [supp_list.index(p) for p in self.properties]
            rst = rst[:, indices]
            convert = [supp_prop[p]["uc"] for p in self.properties]
            extensive = [supp_prop[p]["extensive"] for p in self.properties]
        else:
            convert = [v["uc"] for k, v in supp_prop.items()]
            extensive = [v["extensive"] for k, v in supp_prop.items()]

        if self.unit_conversion:
            rst = np.multiply(rst, convert)

        return rst, extensive
