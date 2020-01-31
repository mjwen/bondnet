"""
The Li-Ec dataset by using molecule level energy, instead of bond energy.
"""

import pandas as pd
import numpy as np
import torch
import logging
from collections import OrderedDict
from rdkit import Chem
from gnn.data.electrolyte import ElectrolyteDataset
from gnn.data.transformers import StandardScaler, GraphFeatureStandardScaler

logger = logging.getLogger(__name__)


class ElectrolyteMoleculeDataset(ElectrolyteDataset):
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

        # if self.pickle_dataset:
        #     if self.pickle_dataset:
        #         if self.grapher == "hetero":
        #             self.save_dataset_hetero()
        #         else:
        #             self.save_dataset()
        #     self.save_state_dict(self._default_state_dict_filename())

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
