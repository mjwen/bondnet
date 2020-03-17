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
from gnn.data.reaction_network import Reaction, ReactionNetwork
from gnn.utils import yaml_load, np_split_by_size
from gnn.data.utils import get_dataset_species


logger = logging.getLogger(__name__)


class ElectrolyteBondDataset(BaseDataset):
    def _load(self):

        logger.info(
            f"Start loading dataset from files: {self.sdf_file}, {self.label_file}, "
            f"and {self.feature_file} ..."
        )

        # read label and feature file
        # TODO, change the label file to a yaml file
        raw_value, raw_indicator, raw_mol_source = self._read_label_file()
        if self.feature_file is not None:
            features = yaml_load(self.feature_file)
        else:
            features = [None] * len(raw_value)

        # build graph for mols from sdf file
        supp = Chem.SDMolSupplier(self.sdf_file, sanitize=True, removeHs=False)
        species = get_dataset_species(self.sdf_file)

        self.graphs = []
        self.labels = []
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

            # TODO make indicator an integer as in BondClassification, and add num_mols.
            #  Then we can combine these dataset. Also, see Reaction dataset.
            label = {
                "value": bonds_energy,  # 1D tensor
                "indicator": bonds_indicator,  # 1D tensor
                "id": bonds_mol_source,  # str
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
                self.labels[i]["label_scaler"] = sca

            logger.info("Label scaler mean: {}".format(mean))
            logger.info("Label scaler std: {}".format(std))

        logger.info("Finish loading {} graphs...".format(len(self.labels)))

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


class ElectrolyteBondDatasetClassification(BaseDataset):
    def __init__(
        self,
        grapher,
        sdf_file,
        label_file,
        feature_file=None,
        feature_transformer=True,
        dtype="float32",
    ):
        super(ElectrolyteBondDatasetClassification, self).__init__(
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
            f"and {self.feature_file} ..."
        )

        # read label and feature file
        raw_value, raw_indicator, raw_mol_source = self._read_label_file()
        if self.feature_file is not None:
            features = yaml_load(self.feature_file)
        else:
            features = [None] * len(raw_value)

        # build graph for mols from sdf file
        supp = Chem.SDMolSupplier(self.sdf_file, sanitize=True, removeHs=False)
        species = get_dataset_species(self.sdf_file)

        self.graphs = []
        self.labels = []
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
            bonds_class = torch.tensor(raw_value[i], dtype=torch.int64)
            bonds_indicator = int(raw_indicator[i])
            bonds_mol_source = raw_mol_source[i]
            label = {
                "value": bonds_class,  # torch.int64
                "indicator": bonds_indicator,  # int
                "id": bonds_mol_source,  # str
            }
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

    def _read_label_file(self):
        value = []
        bond_idx = []
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
                if len(line) > 3:
                    raise ValueError(
                        "Incorrect label file: {}. Expect 3 items per "
                        "line, {} provided.".format(self.label_file, len(line))
                    )

                value.append(int(line[0]))
                bond_idx.append(int(line[1]))
                mol_source.append(line[2])  # it could be a string

        return value, bond_idx, mol_source


class ElectrolyteMoleculeDataset(BaseDataset):
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
            dtype=dtype,
        )

    def _load(self):

        logger.info(
            f"Start loading dataset from files: {self.sdf_file}, {self.label_file}, "
            f"and {self.feature_file} ..."
        )

        # read label and feature file
        raw_labels, extensive = self._read_label_file()
        if self.feature_file is not None:
            features = yaml_load(self.feature_file)
        else:
            features = [None] * len(raw_labels)

        # build graph for mols from sdf file
        supp = Chem.SDMolSupplier(self.sdf_file, sanitize=True, removeHs=False)
        species = get_dataset_species(self.sdf_file)

        self.graphs = []
        self.labels = []
        natoms = []
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
            lb = torch.tensor(lb, dtype=getattr(torch, self.dtype))
            self.labels.append({"value": lb, "id": i})

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
            labels = np.asarray([lb["value"].numpy() for lb in self.labels])
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
            scaled_labels = torch.tensor(
                np.asarray(scaled_labels).T, dtype=getattr(torch, self.dtype)
            )
            transformer_scale = torch.tensor(
                np.asarray(transformer_scale).T, dtype=getattr(torch, self.dtype)
            )

            for i, (lb, ts) in enumerate(zip(scaled_labels, transformer_scale)):
                self.labels[i]["value"] = lb
                self.labels[i]["label_scaler"] = ts

            logger.info("Label scaler mean: {}".format(label_scaler_mean))
            logger.info("Label scaler std: {}".format(label_scaler_std))

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


class ElectrolyteReactionDataset(BaseDataset):
    def _load(self):

        logger.info(
            f"Start loading dataset from files: {self.sdf_file}, {self.label_file}, "
            f"and {self.feature_file} ..."
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

        graphs = []
        for i, (mol, feats) in enumerate(zip(supp, features)):
            if i % 100 == 0:
                logger.info(f"Processing molecule {i}/{len(raw_labels)}")

            if mol is not None:
                g = self.grapher.build_graph_and_featurize(
                    mol, extra_feats_info=feats, dataset_species=species
                )
                # add this for check purpose; some entries in the sdf file may fail
                g.graph_id = i
            else:
                g = None
            graphs.append(g)

        # Should after grapher.build_graph_and_featurize, which initializes the
        # feature name and size
        self._feature_name = self.grapher.feature_name
        self._feature_size = self.grapher.feature_size

        logger.info("Feature name: {}".format(self.feature_name))
        logger.info("Feature size: {}".format(self.feature_size))

        # regroup graphs to reactions
        num_mols = [lb["num_mols"] for lb in raw_labels]
        reactions = np_split_by_size(graphs, num_mols)

        # global feat mapping
        global_mapping = [[{0: 0} for _ in range(n)] for n in num_mols]

        self.graphs = []
        self.labels = []
        for rxn, lb, gmp in zip(reactions, raw_labels, global_mapping):
            if None not in rxn:
                lb["value"] = torch.tensor(lb["value"], dtype=getattr(torch, self.dtype))
                lb["global_mapping"] = gmp
                self.graphs.append(rxn)
                self.labels.append(lb)

        # transformers
        if self.feature_transformer:
            feature_scaler = GraphFeatureStandardScaler()
            graphs = np.concatenate(self.graphs)
            graphs = feature_scaler(graphs)
            num_mols = [len(rxn) for rxn in self.graphs]
            self.graphs = np_split_by_size(graphs, num_mols)
            logger.info("Feature scaler mean: {}".format(feature_scaler.mean))
            logger.info("Feature scaler std: {}".format(feature_scaler.std))

        if self.label_transformer:

            # normalization
            values = [lb["value"] for lb in self.labels]  # list of 0D tensor
            # np and torch compute slightly differently std (depending on `ddof` of np)
            # here we choose to use np
            mean = float(np.mean(values))
            std = float(np.std(values))
            values = (torch.stack(values) - mean) / std
            std = torch.tensor(std, dtype=getattr(torch, self.dtype))

            # update label
            for i, lb in enumerate(values):
                self.labels[i]["value"] = lb
                self.labels[i]["label_scaler"] = std

            logger.info("Label scaler mean: {}".format(mean))
            logger.info("Label scaler std: {}".format(std))

        logger.info("Finish loading {} reactions...".format(len(self.labels)))


class ElectrolyteReactionNetworkDataset(BaseDataset):
    def _load(self):

        logger.info(
            f"Start loading dataset from files: {self.sdf_file}, {self.label_file}, "
            f"and {self.feature_file} ..."
        )

        # read label, feature, and sdf files
        raw_labels = yaml_load(self.label_file)
        if self.feature_file is not None:
            features = yaml_load(self.feature_file)
        else:
            features = [None] * len(raw_labels)
        graphs = build_graphs_from_sdf(
            self.sdf_file, self.grapher, features, len(raw_labels)
        )
        graphs = np.asarray(graphs)
        graphs_not_none_indices = [i for i, g in enumerate(graphs) if g is not None]

        # store feature name and size
        self._feature_name = self.grapher.feature_name
        self._feature_size = self.grapher.feature_size
        logger.info("Feature name: {}".format(self.feature_name))
        logger.info("Feature size: {}".format(self.feature_size))

        # feature transformers
        if self.feature_transformer:
            graphs_not_none = graphs[graphs_not_none_indices]

            feature_scaler = GraphFeatureStandardScaler()
            graphs_not_none = feature_scaler(graphs_not_none)

            # update graphs
            for i, g in zip(graphs_not_none_indices, graphs_not_none):
                graphs[i] = g

            logger.info("Feature scaler mean: {}".format(feature_scaler.mean))
            logger.info("Feature scaler std: {}".format(feature_scaler.std))

        # create reaction
        reactions = []
        self.labels = []
        for i, lb in enumerate(raw_labels):
            mol_ids = lb["reactants"] + lb["products"]
            corrupted = False
            for d in mol_ids:
                if d not in graphs_not_none_indices:
                    corrupted = True
                    break
            # ignore reaction whose reactants or products molecule is None
            if corrupted:
                continue
            rxn = Reaction(
                reactants=lb["reactants"],
                products=lb["products"],
                atom_mapping=lb["atom_mapping"],
                bond_mapping=lb["bond_mapping"],
                id=i,
            )
            reactions.append(rxn)
            label = {
                "value": torch.tensor(lb["value"], dtype=getattr(torch, self.dtype)),
                "id": lb["index"],
            }
            self.labels.append(label)
        self.reaction_ids = list(range(len(reactions)))

        # create reaction network
        self.reaction_network = ReactionNetwork(graphs, reactions)

        if self.label_transformer:

            # normalization
            values = [lb["value"] for lb in self.labels]  # list of 0D tensor
            # np and torch compute slightly differently std (depending on `ddof` of np)
            # here we choose to use np
            mean = float(np.mean(values))
            std = float(np.std(values))
            values = (torch.stack(values) - mean) / std
            std = torch.tensor(std, dtype=getattr(torch, self.dtype))

            # update label
            for i, lb in enumerate(values):
                self.labels[i]["value"] = lb
                self.labels[i]["label_scaler"] = std

            logger.info("Label scaler mean: {}".format(mean))
            logger.info("Label scaler std: {}".format(std))

        logger.info("Finish loading {} reactions...".format(len(self.labels)))

    def __getitem__(self, item):
        rn, rxn, lb = self.reaction_network, self.reaction_ids[item], self.labels[item]
        return rn, rxn, lb

    def __len__(self):
        return len(self.reaction_ids)


def build_graphs_from_sdf(filename, grapher, features, N):

    supp = Chem.SDMolSupplier(filename, sanitize=True, removeHs=False)
    species = get_dataset_species(filename)

    graphs = []
    for i, (mol, feats) in enumerate(zip(supp, features)):
        if i % 100 == 0:
            logger.info(f"Processing molecule {i}/{N}")

        if mol is not None:
            g = grapher.build_graph_and_featurize(
                mol, extra_feats_info=feats, dataset_species=species
            )
            # add this for check purpose; some entries in the sdf file may fail
            g.graph_id = i
        else:
            g = None

        graphs.append(g)

    return graphs
