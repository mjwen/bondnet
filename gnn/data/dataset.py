import numpy as np
from collections import defaultdict
from rdkit import Chem
from gnn.utils import expand_path


class BaseDataset:
    """
    Base dataset class.

   Args:
    grapher (BaseGraph): grapher object that build different types of graphs:
        `hetero`, `homo_bidirected` and `homo_complete`.
        For hetero graph, atom, bond, and global state are all represented as
        graph nodes. For homo graph, atoms are represented as node and bond are
        represented as graph edges.
    molecules (list or str): rdkit molecules. If a string, it should be the path
        to the sdf file of the molecules.
    labels (list or str): each element is a dict representing the label for a bond,
        molecule or reaction. If a string, it should be the path to the label file.
    extra_features (list or str or None): each element is a dict representing extra
        features provided to the molecules. If a string, it should be the path to the
        feature file. If `None`, features will be calculated only using rdkit.
    feature_transformer (bool): If `True`, standardize the features by subtracting the
        means and then dividing the standard deviations.
    label_transformer (bool): If `True`, standardize the label by subtracting the
        means and then dividing the standard deviations. More explicitly,
        labels are standardized by y' = (y - mean(y))/std(y), the model will be
        trained on this scaled value. However for metric measure (e.g. MAE) we need
        to convert y' back to y, i.e. y = y' * std(y) + mean(y), the model
        prediction is then y^ = y'^ *std(y) + mean(y), where ^ means predictions.
        Then MAE is |y^-y| = |y'^ - y'| *std(y), i.e. we just need to multiple
        standard deviation to get back to the original scale. Similar analysis
        applies to RMSE.
    state_dict_filename (str or None): If `None`, feature mean and std (if
        feature_transformer is True) and label mean and std (if label_transformer is True)
        are computed from the dataset; otherwise, they are read from the file.
    """

    def __init__(
        self,
        grapher,
        molecules,
        labels,
        extra_features=None,
        feature_transformer=True,
        label_transformer=True,
        dtype="float32",
        state_dict_filename=None,
    ):

        if dtype not in ["float32", "float64"]:
            raise ValueError(f"`dtype {dtype}` should be `float32` or `float64`.")

        self.grapher = grapher
        self.molecules = (
            expand_path(molecules) if isinstance(molecules, str) else molecules
        )
        self.raw_labels = expand_path(labels) if isinstance(labels, str) else labels
        self.extra_features = (
            expand_path(extra_features)
            if isinstance(extra_features, str)
            else extra_features
        )
        self.feature_transformer = feature_transformer
        self.label_transformer = label_transformer
        self.dtype = dtype
        self.state_dict_filename = state_dict_filename

        self.graphs = None
        self.labels = None
        self._feature_size = None
        self._feature_name = None
        self._feature_scaler_mean = None
        self._feature_scaler_std = None
        self._label_scaler_mean = None
        self._label_scaler_std = None
        self._species = None
        self._failed = None

        self._load()

    @property
    def feature_size(self):
        """
        Returns a dict of feature size with node type as the key.
        """
        return self._feature_size

    @property
    def feature_name(self):
        """
        Returns a dict of feature name with node type as the key.
        """
        return self._feature_name

    def get_feature_size(self, ntypes):
        """
        Get feature sizes.

        Args:
              ntypes (list of str): types of nodes.

        Returns:
             list: sizes of features corresponding to note types in `ntypes`.
        """
        size = []
        for nt in ntypes:
            for k in self.feature_size:
                if nt in k:
                    size.append(self.feature_size[k])
        # TODO more checks needed e.g. one node get more than one size
        msg = f"cannot get feature size for nodes: {ntypes}"
        assert len(ntypes) == len(size), msg

        return size

    @property
    def failed(self):
        """
        Whether an entry (molecule, reaction) fails upon converting using rdkit.

        Returns:
            list of bool: each element indicates whether a entry fails. The size of
                this list is the same as the labels, each one corresponds a label in
                the same order.
            None: is this info is not set
        """
        return self._failed

    def state_dict(self):
        d = {
            "feature_size": self._feature_size,
            "feature_name": self._feature_name,
            "feature_scaler_mean": self._feature_scaler_mean,
            "feature_scaler_std": self._feature_scaler_std,
            "label_scaler_mean": self._label_scaler_mean,
            "label_scaler_std": self._label_scaler_std,
            "species": self._species,
        }

        return d

    def load_state_dict(self, d):
        self._feature_size = d["feature_size"]
        self._feature_name = d["feature_name"]
        self._feature_scaler_mean = d["feature_scaler_mean"]
        self._feature_scaler_std = d["feature_scaler_std"]
        self._label_scaler_mean = d["label_scaler_mean"]
        self._label_scaler_std = d["label_scaler_std"]
        self._species = d["species"]

    def _load(self):
        """Read data from files and then featurize."""
        raise NotImplementedError

    @staticmethod
    def get_molecules(molecules):
        if isinstance(molecules, str):
            supp = Chem.SDMolSupplier(molecules, sanitize=True, removeHs=False)
            molecules = [m for m in supp]
        return molecules

    @staticmethod
    def build_graphs(grapher, molecules, features, species):
        """
        Build DGL graphs using grapher for the molecules.

        Args:
            grapher (Grapher): grapher object to create DGL graphs
            molecules (list): rdkit molecules
            features (list): each element is a dict of extra features for a molecule
            species (list): chemical species (str) in all molecules

        Returns:
            list: DGL graphs
        """

        graphs = []
        for i, (m, feats) in enumerate(zip(molecules, features)):
            if m is not None:
                g = grapher.build_graph_and_featurize(
                    m, extra_feats_info=feats, dataset_species=species
                )
                # add this for check purpose; some entries in the sdf file may fail
                g.graph_id = i
            else:
                g = None

            graphs.append(g)

        return graphs

    def __getitem__(self, item):
        """Get datapoint with index

        Args:
            item (int): data point index

        Returns:
            g (DGLGraph or DGLHeteroGraph): graph ith data point
            lb (dict): Labels of the data point
        """
        g, lb, = self.graphs[item], self.labels[item]
        return g, lb

    def __len__(self):
        """Length of the dataset.

        Returns:
            int: length of dataset
        """
        return len(self.graphs)

    def __repr__(self):
        rst = "Dataset " + self.__class__.__name__ + "\n"
        rst += "Length: {}\n".format(len(self))
        for ft, sz in self.feature_size.items():
            rst += "Feature: {}, size: {}\n".format(ft, sz)
        for ft, nm in self.feature_name.items():
            rst += "Feature: {}, name: {}\n".format(ft, nm)
        return rst


class Subset(BaseDataset):
    def __init__(self, dataset, indices):
        self.dtype = dataset.dtype
        self.dataset = dataset
        self.indices = indices

    @property
    def feature_size(self):
        return self.dataset.feature_size

    @property
    def feature_name(self):
        return self.dataset.feature_name

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def train_validation_test_split(dataset, validation=0.1, test=0.1, random_seed=None):
    """
    Split a dataset into training, validation, and test set.

    The training set will be automatically determined based on `validation` and `test`,
    i.e. train = 1 - validation - test.

    Args:
        dataset: the dataset
        validation (float, optional): The amount of data (fraction) to be assigned to
            validation set. Defaults to 0.1.
        test (float, optional): The amount of data (fraction) to be assigned to test
            set. Defaults to 0.1.
        random_seed (int, optional): random seed that determines the permutation of the
            dataset. Defaults to 35.

    Returns:
        [train set, validation set, test_set]
    """
    assert validation + test < 1.0, "validation + test >= 1"
    size = len(dataset)
    num_val = int(size * validation)
    num_test = int(size * test)
    num_train = size - num_val - num_test

    if random_seed is not None:
        np.random.seed(random_seed)
    idx = np.random.permutation(size)
    train_idx = idx[:num_train]
    val_idx = idx[num_train : num_train + num_val]
    test_idx = idx[num_train + num_val :]
    return [
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    ]


def train_validation_test_split_test_with_all_bonds_of_mol(
    dataset, validation=0.1, test=0.1, random_seed=None
):
    """
    Split a dataset into training, validation, and test set.

    Different from `train_validation_test_split`, where the split of dataset is bond
    based, here the bonds from a molecule either goes to (train, validation) set or
    test set. This is used to evaluate the prediction order of bond energy.

    The training set will be automatically determined based on `validation` and `test`,
    i.e. train = 1 - validation - test.

    Args:
        dataset: the dataset
        validation (float, optional): The amount of data (fraction) to be assigned to
            validation set. Defaults to 0.1.
        test (float, optional): The amount of data (fraction) to be assigned to test
            set. Defaults to 0.1.
        random_seed (int, optional): random seed that determines the permutation of the
            dataset. Defaults to 35.

    Returns:
        [train set, validation set, test_set]
    """
    assert validation + test < 1.0, "validation + test >= 1"
    size = len(dataset)
    num_val = int(size * validation)
    num_test = int(size * test)
    num_train = size - num_val - num_test

    # group by molecule
    groups = defaultdict(list)
    for i, (_, label) in enumerate(dataset):
        groups[label["id"]].append(i)
    groups = [val for key, val in groups.items()]

    # permute on the molecule level
    if random_seed is not None:
        np.random.seed(random_seed)
    idx = np.random.permutation(len(groups))
    test_idx = []
    train_val_idx = []
    for i in idx:
        if len(test_idx) < num_test:
            test_idx.extend(groups[i])
        else:
            train_val_idx.extend(groups[i])

    # permute on the bond level for train and validation
    idx = np.random.permutation(train_val_idx)
    train_idx = idx[:num_train]
    val_idx = idx[num_train:]

    return [
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    ]
