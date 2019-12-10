"""
QM9 dataset.
"""
# pylint: disable=not-callable,no-member


import torch
import logging
from rdkit import Chem
from rdkit.Chem.rdchem import GetPeriodicTable
from gnn.data.featurizer import (
    BaseFeaturizer,
    AtomFeaturizer,
    BondFeaturizer,
    HeteroMoleculeGraph,
)
from gnn.data.electrolyte import ElectrolyteDataset
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class GlobalStateFeaturizer(BaseFeaturizer):
    """
    Featurize the global state of a molecules.
    """

    def __init__(self, dtype="float32"):
        super(GlobalStateFeaturizer, self).__init__(dtype)
        self._feature_size = None
        self._feature_name = None

    @property
    def feature_size(self):
        return self._feature_size

    @property
    def feature_name(self):
        return self._feature_name

    def __call__(self, mol):

        global_feats_dict = dict()
        pd = GetPeriodicTable()
        g = [
            mol.GetNumAtoms(),
            mol.GetNumBonds(),
            sum([pd.GetAtomicWeight(a.GetAtomicNum()) for a in mol.GetAtoms()]),
        ]

        dtype = getattr(torch, self.dtype)
        global_feats_dict["feat"] = torch.tensor([g], dtype=dtype)

        self._feature_size = len(g)
        self._feature_name = ["num atoms", "num bonds", "molecule weight"]

        return global_feats_dict


class QM9Dataset(ElectrolyteDataset):
    """
    The QM9 dataset.
    """

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
            bond_featurizer = BondFeaturizer(dtype=self.dtype)
            global_featurizer = GlobalStateFeaturizer(dtype=self.dtype)

            labels = self._read_label_file()

            supp = Chem.SDMolSupplier(self.sdf_file, sanitize=True, removeHs=False)

            self.graphs = []
            bad_mols = []
            for i, mol in enumerate(supp):
                if i % 100 == 0:
                    logger.info("Processing molecule {}/{}".format(i, len(labels)))

                if mol is None:
                    bad_mols.append(i)
                    continue

                grapher = HeteroMoleculeGraph(
                    atom_featurizer=atom_featurizer,
                    bond_featurizer=bond_featurizer,
                    global_state_featurizer=global_featurizer,
                    self_loop=self.self_loop,
                )
                g = grapher.build_graph_and_featurize(mol)
                self.graphs.append(g)

            labels = np.delete(labels, bad_mols, axis=0)
            self.labels = torch.tensor(labels, dtype=getattr(torch, self.dtype))

            self._feature_size = {
                "atom": atom_featurizer.feature_size,
                "bond": bond_featurizer.feature_size,
                "global": global_featurizer.feature_size,
            }
            self._feature_name = {
                "bond": bond_featurizer.feature_name,
                "atom": atom_featurizer.feature_name,
                "global": global_featurizer.feature_name,
            }

            if self.pickle_dataset:
                self.save_dataset()
                filename = self._default_state_dict_filename()
                self.save_state_dict(filename)

        logger.info("Finish loading {} graphs...".format(len(self.labels)))

    def _read_label_file(self):
        rst = pd.read_csv(self.label_file, index_col=0)
        rst = rst.to_numpy()
        return rst
