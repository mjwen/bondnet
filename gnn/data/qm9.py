"""
The Li-EC electrolyte dataset.
"""
# pylint: disable=not-callable,no-member

import torch
import logging
from gnn.utils import pickle_load
from gnn.data.featurizer import (
    AtomFeaturizer,
    BondFeaturizer,
    GlobalStateFeaturizer,
    HeteroMoleculeGraph,
)
from gnn.data.electrolyte import ElectrolyteDataset
import pandas as pd
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import rdmolops
    from rdkit.Chem.rdmolops import FastFindRings
except ImportError:
    pass


logger = logging.getLogger(__name__)


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

            self.graphs = pickle_load(self.sdf_file)
            self.labels = pickle_load(self.label_file)
            filename = self._default_state_dict_filename()
            d = self.load_state_dict(filename)
            self._feature_size = d["feature_size"]
            self._feature_name = d["feature_name"]

        else:
            logger.info(
                "Start loading dataset from files {} and {}...".format(
                    self.sdf_file, self.label_file
                )
            )

            species, bad_mols = self._get_species()
            atom_featurizer = AtomFeaturizer(species, dtype=self.dtype)
            bond_featurizer = BondFeaturizer(dtype=self.dtype)
            global_featurizer = GlobalStateFeaturizer(dtype=self.dtype)

            labels = np.delete(self._read_label_file(), bad_mols, axis=0)
            self.labels = torch.tensor(labels, dtype=getattr(torch, self.dtype))
            dataset_size = len(self.labels)

            self.graphs = []
            supp = Chem.SDMolSupplier(self.sdf_file, sanitize=True, removeHs=False)
            for i, mol in enumerate(supp):
                if i % 100 == 0:
                    logger.info("Processing molecule {}/{}".format(i, dataset_size))

                if i in bad_mols:  # skip bad mols
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
                filename = self._default_state_dict_filename()
                self.save_state_dict(filename)

        logger.info("Finish loading {} graphs...".format(len(self.labels)))

    def _read_label_file(self):
        rst = pd.read_csv(self.label_file, index_col=0)
        rst = rst.to_numpy()
        return rst

    def _get_species(self):
        suppl = Chem.SDMolSupplier(self.sdf_file, sanitize=True, removeHs=False)
        system_species = set()
        bad_mols = []
        for i, mol in enumerate(suppl):
            try:
                atoms = mol.GetAtoms()
                species = [a.GetSymbol() for a in atoms]
                system_species.update(species)
            except AttributeError:
                bad_mols.append(i)
                logger.info("Bad mol '{}' from sdf file.".format(i))
        return list(system_species), bad_mols
