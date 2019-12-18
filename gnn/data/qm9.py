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
from gnn.data.utils import StandardScaler

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
        normalize_extensive (bool):
        pickle_dataset:
        dtype:
    """

    def __init__(
        self,
        sdf_file,
        label_file,
        self_loop=True,
        grapher="hetero",
        bond_length_featurizer=None,
        properties=None,
        unit_conversion=True,
        normalize_extensive=True,
        pickle_dataset=False,
        dtype="float32",
    ):

        self.properties = properties
        self.unit_conversion = unit_conversion
        self.normalize_extensive = normalize_extensive
        super(QM9Dataset, self).__init__(
            sdf_file,
            label_file,
            self_loop,
            grapher,
            bond_length_featurizer,
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

            d = self.load_state_dict(self._default_state_dict_filename())
            self._feature_size = d["feature_size"]
            self._feature_name = d["feature_name"]

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
            self.labels = []
            supp = Chem.SDMolSupplier(self.sdf_file, sanitize=True, removeHs=False)

            for i, mol in enumerate(supp):
                if i % 100 == 0:
                    logger.info("Processing molecule {}/{}".format(i, len(raw_labels)))

                if mol is None:
                    continue

                g = grapher.build_graph_and_featurize(mol)
                self.graphs.append(g)

                # normalize extensive properties
                natoms = mol.GetNumAtoms()
                normalizer = [natoms if t else 1.0 for t in extensive]
                lb = raw_labels[i]
                if self.normalize_extensive:
                    lb = np.divide(lb, normalizer)
                self.labels.append(lb)

            self.labels = torch.tensor(self.labels, dtype=getattr(torch, self.dtype))

            # standardize features
            scaler = StandardScaler()
            self.graphs = scaler(self.graphs)
            logger.info("StandardScaler mean: {}".format(scaler.mean))
            logger.info("StandardScaler std: {}".format(scaler.std))

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
                filename = self._default_state_dict_filename()
                self.save_state_dict(filename)

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
