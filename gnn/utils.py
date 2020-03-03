import os
import time
import pickle
import yaml
import random
import numpy as np
import torch
import dgl
import logging
import warnings
import sys
import numpy as np

logger = logging.getLogger(__name__)


def np_split_by_size(array, sizes, axis=0):
    """
    Split an array into `len(sizes)` chunks with size of each chunk in dim
    according to `sizes`.

    This is a convenient function over np.split(), where one need to give the indices at
    which to split the array (not easy to work with).

    Args:
        array:
        sections (list): size of each chunk.
        axis (int): the axis along which to split the data

    Returns:
        list: a list of array.

    Example:
        >>> np_split_by_size([0,1,2,3,4,5], [1,2,3])
        >>>[[0], [1,2], [3,4,5]]
    """
    array = np.asarray(array)
    assert array.shape[axis] == sum(sizes), "array.shape[axis] not equal to sum(sizes)"

    tot = 0
    indices = []
    for i in sizes:
        tot += i
        indices.append(tot)
    indices = indices[:-1]
    return np.split(array, indices, axis=axis)


def expand_path(path):
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def create_directory(filename):
    filename = expand_path(filename)
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return filename


def pickle_dump(obj, filename):
    filename = expand_path(filename)
    create_directory(filename)
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(filename):
    filename = expand_path(filename)
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj


def yaml_dump(obj, filename):
    filename = expand_path(filename)
    create_directory(filename)
    with open(filename, "w") as f:
        yaml.dump(obj, f, default_flow_style=False)


def yaml_load(filename):
    filename = expand_path(filename)
    with open(filename, "r") as f:
        obj = yaml.safe_load(f)
    return obj


def stat_cuda(msg):
    print("-" * 10, "cuda status:", msg, "-" * 10)
    print(
        "allocated: {}M, max allocated: {}M, cached: {}M, max cached: {}M".format(
            torch.cuda.memory_allocated() / 1024 / 1024,
            torch.cuda.max_memory_allocated() / 1024 / 1024,
            torch.cuda.memory_cached() / 1024 / 1024,
            torch.cuda.max_memory_cached() / 1024 / 1024,
        )
    )


def seed_torch(seed=35):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dgl.random.seed(seed)


def save_checkpoints(objects, msg=None):
    """
    Save checkpoints for all objects for later recovery.

    Args:
        objects (dict): A dictionary of objects to save. The keys are identifier to the
            objects.
        msg (str): a message to log.

    """

    m = "Save checkpoints: "
    for k, obj in objects.items():
        filename = "{}_checkpoint.pkl".format(k)
        torch.save(obj.state_dict(), filename)
        m += "{}, ".format(filename)
    if msg is not None:
        m += msg
    logger.info(m)


def load_checkpoints(objects):
    """
    Load checkpoints for all objects for later recovery.

    Args:
        objects (dict): A dictionary of objects to save. The keys are identifier to the
        objects.
    """
    for k, obj in objects.items():
        filename = "{}_checkpoint.pkl".format(k)
        obj.load_state_dict(torch.load(filename))
        logger.info("Load checkpoints {}".format(filename))


class Timer:
    def __init__(self):
        self.first = None
        self.latest = None

    def step(self, msg=None):
        if self.first is None:
            self.first = self.latest = time.time()
        current = time.time()
        if msg is None:
            m = ""
        else:
            m = " User message: {}.".format(msg)
        print(
            "{:.2f} | {:.2f}. Time (s) since last called and since first called.{}".format(
                current - self.latest, current - self.first, m
            )
        )
        self.latest = current


def warn_stdout(message, category, filename, lineno, file=None, line=None):
    """
    Redirect warning message to stdout instead of stderr.

    To use this:
    >>> warnings.showwarning = warn_stdout
    >>> warnings.warn("some warning message")

    see: https://stackoverflow.com/questions/858916/how-to-redirect-python-warnings-to-a-custom-stream
    """
    sys.stdout.write(warnings.formatwarning(message, category, filename, lineno))
