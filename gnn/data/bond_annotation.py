import torch
import logging
from rdkit import Chem
from gnn.data.dataset import BaseDataset
from gnn.data.transformers import GraphFeatureStandardScaler
from gnn.utils import yaml_load
from gnn.data.utils import get_dataset_species


logger = logging.getLogger(__name__)


class BondAnnotationDataset(BaseDataset):
    def __init__(
        self,
        grapher,
        sdf_file,
        label_file,
        feature_file=None,
        feature_transformer=True,
        dtype="float32",
    ):
        super(BondAnnotationDataset, self).__init__(
            grapher=grapher,
            sdf_file=sdf_file,
            label_file=label_file,
            feature_file=feature_file,
            feature_transformer=feature_transformer,
            label_transformer=False,
            dtype=dtype,
        )

    def _load(self):

        logger.info(
            f"Start loading dataset from files: {self.sdf_file}, {self.label_file}, "
            f"and {self.feature_file}..."
        )

        # read label and feature file
        raw_labels = yaml_load(self.label_file)
        if self.feature_file is not None:
            features = yaml_load(self.feature_file)
        else:
            features = [None] * len(raw_labels)

        # build graph for mols from sdf file
        supp = Chem.SDMolSupplier(self.sdf_file, sanitize=True, removeHs=False)
        species = get_dataset_species(self.sdf_file)

        self.graphs = []
        self.labels = []
        for i, mol in enumerate(supp):
            if i % 100 == 0:
                logger.info("Processing molecule {}/{}".format(i, len(raw_labels)))

            # bad mol
            if mol is None:
                continue

            # graph
            g = self.grapher.build_graph_and_featurize(
                mol, extra_feats_info=features[i], dataset_species=species
            )
            # add this for check purpose; some entries in the sdf file may fail
            g.graph_id = i
            self.graphs.append(g)

            # label
            bonds_class = torch.tensor(raw_labels[i], dtype=torch.int64)
            label = {"value": bonds_class, "id": i}
            self.labels.append(label)

        # Should after grapher.build_graph_and_featurize, which initializes the
        # feature name and size
        self._feature_name = self.grapher.feature_name
        self._feature_size = self.grapher.feature_size

        logger.info("Feature name: {}".format(self.feature_name))
        logger.info("Feature size: {}".format(self.feature_size))

        # feature transformers
        if self.feature_transformer:
            feature_scaler = GraphFeatureStandardScaler()
            self.graphs = feature_scaler(self.graphs)

            logger.info("Feature scaler mean: {}".format(feature_scaler.mean))
            logger.info("Feature scaler std: {}".format(feature_scaler.std))

        logger.info("Finish loading {} graphs...".format(len(self.labels)))
