"""
QM9 dataset.
"""

import torch
import logging
from collections import OrderedDict
from rdkit import Chem
from gnn.data.featurizer import (
    AtomFeaturizer,
    BondAsNodeFeaturizer,
    BondAsEdgeBidirectedFeaturizer,
    MolWeightFeaturizer,
)
from gnn.data.grapher import HomoBidirectedGraph, HeteroMoleculeGraph
from gnn.data.electrolyte import ElectrolyteDataset
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class QM9Dataset(ElectrolyteDataset):
    """
    The QM9 dataset.
    """

    def __init__(
        self,
        sdf_file,
        label_file,
        self_loop=True,
        hetero=True,
        properties=None,
        unit_conversion=True,
        pickle_dataset=False,
        dtype="float32",
    ):
        self.properties = properties
        self.unit_conversion = unit_conversion
        super(QM9Dataset, self).__init__(
            sdf_file, label_file, self_loop, hetero, pickle_dataset, dtype
        )

    def _load(self):
        if self._pickled:
            logger.info(
                "Start loading dataset from picked files {} and {}...".format(
                    self.sdf_file, self.label_file
                )
            )

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
                bond_featurizer = BondAsNodeFeaturizer(dtype=self.dtype)
                global_featurizer = MolWeightFeaturizer(dtype=self.dtype)
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

            labels = self._read_label_file()

            self.graphs = []
            bad_mols = []
            supp = Chem.SDMolSupplier(self.sdf_file, sanitize=True, removeHs=False)

            for i, mol in enumerate(supp):
                if i % 100 == 0:
                    logger.info("Processing molecule {}/{}".format(i, len(labels)))

                if mol is None:
                    bad_mols.append(i)
                    continue

                g = grapher.build_graph_and_featurize(mol)
                self.graphs.append(g)

            labels = np.delete(labels, bad_mols, axis=0)
            self.labels = torch.tensor(labels, dtype=getattr(torch, self.dtype))

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
                self.save_dataset()
                filename = self._default_state_dict_filename()
                self.save_state_dict(filename)

        logger.info("Finish loading {} graphs...".format(len(self.labels)))

    def _read_label_file(self):

        rst = pd.read_csv(self.label_file, index_col=0)
        rst = rst.to_numpy()

        h2e = 27.211396132  # Hatree to eV
        k2e = 0.0433634  # kcal/mol to eV

        # supported property and unit conversion
        supp_prop = OrderedDict()
        supp_prop["A"] = 1.0
        supp_prop["B"] = 1.0
        supp_prop["C"] = 1.0
        supp_prop["mu"] = 1.0
        supp_prop["alpha"] = 1.0
        supp_prop["homo"] = h2e
        supp_prop["lumo"] = h2e
        supp_prop["gap"] = h2e
        supp_prop["r2"] = 1.0
        supp_prop["zpve"] = h2e
        supp_prop["u0"] = h2e
        supp_prop["u298"] = h2e
        supp_prop["h298"] = h2e
        supp_prop["g298"] = h2e
        supp_prop["cv"] = 1
        supp_prop["u0_atom"] = k2e
        supp_prop["u298_atom"] = k2e
        supp_prop["h298_atom"] = k2e
        supp_prop["g298_atom"] = k2e

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
            convert = [supp_prop[p] for p in self.properties]
        else:
            convert = [v for k, v in supp_prop.items()]

        if self.unit_conversion:
            rst = np.multiply(rst, convert)

        return rst
