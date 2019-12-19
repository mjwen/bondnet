"""
QM9 dataset.
"""

import pandas as pd
import numpy as np
import torch
import logging
from collections import OrderedDict
from rdkit import Chem
from gnn.data.featurizer import (
    AtomFeaturizer,
    BondAsNodeFeaturizer,
    BondAsEdgeBidirectedFeaturizer,
    BondAsEdgeCompleteFeaturizer,
    MolWeightFeaturizer,
)
from gnn.data.grapher import HomoBidirectedGraph, HomoCompleteGraph, HeteroMoleculeGraph
from gnn.data.electrolyte import ElectrolyteDataset
from gnn.data.transformers import StandardScaler, GraphFeatureStandardScaler

logger = logging.getLogger(__name__)


class QM9Dataset(ElectrolyteDataset):
    """
    The QM9 dataset.

    Args:
        sdf_file:
        label_file:
        self_loop:
        grapher (str): the type of graph to create, options are: `hetero`,
            `homo_bidirected` and `homo_complete`.
        properties (list of str): the dataset propery to use. If `None`, use all.
        unit_conversion (bool):
        feature_transformer (bool):
        label_transformer (bool):
    """

    def __init__(
        self,
        sdf_file,
        label_file,
        self_loop=True,
        grapher="hetero",
        bond_length_featurizer=None,
        feature_transformer=True,
        label_transformer=True,
        properties=None,
        unit_conversion=True,
        pickle_dataset=False,
        dtype="float32",
    ):

        self.properties = properties
        self.unit_conversion = unit_conversion
        super(QM9Dataset, self).__init__(
            sdf_file,
            label_file,
            self_loop,
            grapher,
            bond_length_featurizer,
            feature_transformer,
            label_transformer,
            pickle_dataset,
            dtype,
        )

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
            atom_featurizer = AtomFeaturizer(species, dtype=self.dtype)
            if self.grapher == "hetero":
                bond_featurizer = BondAsNodeFeaturizer(
                    length_featurizer=self.bond_length_featurizer, dtype=self.dtype
                )
                global_featurizer = MolWeightFeaturizer(dtype=self.dtype)
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

            # read mol graphs and label
            raw_labels, extensive = self._read_label_file()
            self.graphs = []
            labels = []
            supp = Chem.SDMolSupplier(self.sdf_file, sanitize=True, removeHs=False)

            natoms = []
            for i, mol in enumerate(supp):
                if i % 100 == 0:
                    logger.info("Processing molecule {}/{}".format(i, len(raw_labels)))

                if mol is None:
                    continue

                g = grapher.build_graph_and_featurize(mol)
                self.graphs.append(g)
                labels.append(raw_labels[i])
                natoms.append(mol.GetNumAtoms())

            # feature and lable transformer
            if self.feature_transformer:
                feature_scaler = GraphFeatureStandardScaler()
                self.graphs = feature_scaler(self.graphs)
                logger.info("Feature scaler mean: {}".format(feature_scaler.mean))
                logger.info("Feature scaler std: {}".format(feature_scaler.std))

            if self.label_transformer:
                labels = np.asarray(labels)
                natoms = np.asarray(natoms, dtype=np.float32)

                # # intensive labels standardized by y' = (y - mean(y))/std(y)
                # int_indices = [i for i, t in enumerate(extensive) if not t]
                # int_labels = labels[:, int_indices]
                # int_scaler = GraphFeatureStandardScaler()
                # int_labels = int_scaler(int_labels)
                # int_ts = np.repeat([int_scaler.std], len(labels), axis=0)
                #
                # # extensive labels standardized by the number of atoms in the molecule
                # # i.e. y' = y/natoms, i.e. by
                # ext_indices = [i for i, t in enumerate(extensive) if t]
                # ext_labels = labels[:, ext_indices]
                # ext_ts = np.repeat([natoms], len(ext_indices), axis=0).T
                # ext_labels /= ext_ts
                #
                # # combine scaled labels and create transformer scaler
                # ii = 0
                # ei = 0
                # labels = []
                # transformer_scale = []
                # for ci in range(len(extensive)):
                #     if ci in int_indices:
                #         labels.append(int_labels[:, ii])
                #         transformer_scale.append(int_ts[:, ii])
                #         ii += 1
                #     if ci in ext_indices:
                #         labels.append(ext_labels[:, ei])
                #         transformer_scale.append(ext_ts[:, ei])
                #         ei += 1
                #     else:
                #         raise RuntimeError("indices not found. this should never happen")
                # labels = np.asarray(labels).T
                # self.transformer_scale = torch.tensor(
                #     np.asarray(transformer_scale).T, dtype=getattr(torch, self.dtype)
                # )

                # this is equivalent to the above one, but simpler

                scaled_labels = []
                transformer_scale = []
                for i, is_ext in enumerate(extensive):
                    if is_ext:
                        # extensive labels standardized by the number of atoms in the
                        # molecules, i.e. y' = y/natoms
                        lb = labels[:, i]
                        lb /= natoms
                        ts = natoms
                    else:
                        # intensive labels standardized by y' = (y - mean(y))/std(y)
                        scaler = StandardScaler()
                        lb = torch.from_numpy(labels[:, [i]])  # 2D array of shape (N, 1)
                        lb = scaler(lb)
                        lb = lb.numpy().ravel()
                        ts = np.repeat(scaler.std.numpy(), len(lb))
                    scaled_labels.append(lb)
                    transformer_scale.append(ts)
                labels = np.asarray(scaled_labels).T
                self.transformer_scale = torch.tensor(
                    np.asarray(transformer_scale).T, dtype=getattr(torch, self.dtype)
                )

            self.labels = torch.tensor(labels, dtype=getattr(torch, self.dtype))

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
                if self.pickle_dataset:
                    if self.grapher == "hetero":
                        self.save_dataset_hetero()
                    else:
                        self.save_dataset()
                self.save_state_dict(self._default_state_dict_filename())

        logger.info("Finish loading {} graphs...".format(len(self.labels)))

    def _read_label_file(self):

        rst = pd.read_csv(self.label_file, index_col=0)
        rst = rst.to_numpy()

        h2e = 27.211396132  # Hatree to eV
        k2e = 0.0433634  # kcal/mol to eV

        # supported property
        supp_prop = OrderedDict()
        supp_prop["A"] = {"uc": 1.0, "extensive": True}
        supp_prop["B"] = {"uc": 1.0, "extensive": True}
        supp_prop["C"] = {"uc": 1.0, "extensive": True}
        supp_prop["mu"] = {"uc": 1.0, "extensive": True}
        supp_prop["alpha"] = {"uc": 1.0, "extensive": True}
        supp_prop["homo"] = {"uc": h2e, "extensive": False}
        supp_prop["lumo"] = {"uc": h2e, "extensive": False}
        supp_prop["gap"] = {"uc": h2e, "extensive": False}
        supp_prop["r2"] = {"uc": 1.0, "extensive": True}
        supp_prop["zpve"] = {"uc": h2e, "extensive": True}
        supp_prop["u0"] = {"uc": h2e, "extensive": True}
        supp_prop["u298"] = {"uc": h2e, "extensive": True}
        supp_prop["h298"] = {"uc": h2e, "extensive": True}
        supp_prop["g298"] = {"uc": h2e, "extensive": True}
        supp_prop["cv"] = {"uc": 1.0, "extensive": True}
        supp_prop["u0_atom"] = {"uc": k2e, "extensive": True}
        supp_prop["u298_atom"] = {"uc": k2e, "extensive": True}
        supp_prop["h298_atom"] = {"uc": k2e, "extensive": True}
        supp_prop["g298_atom"] = {"uc": k2e, "extensive": True}

        if self.properties is not None:

            for prop in self.properties:
                if prop not in supp_prop:
                    raise ValueError(
                        "Property '{}' not supported. Supported ones are: {}".format(
                            property, supp_prop
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
