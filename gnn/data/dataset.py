import numpy as np


class BaseDataset:
    """
    Base dataset class.
    """

    def __init__(self, dtype="float32"):
        if dtype not in ["float32", "float64"]:
            raise ValueError(
                "`dtype` should be `float32` or `float64`, but got `{}`.".format(dtype)
            )
        self.dtype = dtype
        self.graphs = None
        self.labels = None
        self.transformer_scale = None

    @property
    def feature_size(self):
        """
        Returns a dict of feature size with node type as the key.
        """
        raise NotImplementedError

    @property
    def feature_name(self):
        """
        Returns a dict of feature name with node type as the key.
        """
        raise NotImplementedError

    def get_feature_size(self, ntypes):
        """
        Returns a list of the feature corresponding to the note types `ntypes`.
        """
        size = []
        for n in ntypes:
            for k in self.feature_size:
                if n in k:
                    size.append(self.feature_size[k])
        # TODO more checks needed e.g. one node get more than one size
        msg = "cannot get feature size for nodes: {}".format(ntypes)
        assert len(ntypes) == len(size), msg
        return size

    def __getitem__(self, item):
        """Get datapoint with index

        Args:
            item (int): Datapoint index

        Returns:
            g: DGLHeteroGraph for the ith datapoint
            lb (dict): Labels of the datapoint
            s (float or array): transformer scaler that is supposed to be multiplied by
                the difference between the label and the model prediction, after which
                the difference should get back to the original scale of the label.
                Should have the same shape as `value` of label.
                For example, suppose labels are standardized by y' = (y - mean(y))/std(y),
                the model will be trained on this scaled value.
                However for metric measure (e.g. MAE) we need to convert y' back to y,
                i.e. y = y' * std(y) + mean(y), the model prediction is then
                y^ = y'^ *std(y) + mean(y), where ^ means predictions.
                Then MAE is |y^-y| = |y'^ - y'| *std(y), i.e. we just need to multiple
                standard deviation to get back to the original scale. Similar analysis
                applies to RMSE.
        """
        g, lb, = self.graphs[item], self.labels[item]
        s = None if self.transformer_scale is None else self.transformer_scale[item]
        return g, lb, s

    def __len__(self):
        """Length of the dataset

        Returns:
            Length of Dataset
        """
        return len(self.graphs)


class Subset(BaseDataset):
    def __init__(self, dataset, indices):
        super(Subset, self).__init__()
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


def train_validation_test_split(dataset, validation=0.1, test=0.1, random_seed=35):
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
