import os
import time
import pickle
import yaml
import random
import torch
import dgl
import logging
import warnings
import sys
import shutil
import itertools
import copy
from pathlib import Path
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
        sizes (list): size of each chunk.
        axis (int): the axis along which to split the data

    Returns:
        list: a list of array.

    Example:
        >>> np_split_by_size([0,1,2,3,4,5], [1,2,3])
        >>>[[0], [1,2], [3,4,5]]
    """
    array = np.asarray(array)
    assert array.shape[axis] == sum(sizes), "array.shape[axis] not equal to sum(sizes)"

    indices = list(itertools.accumulate(sizes))
    indices = indices[:-1]

    return np.split(array, indices, axis=axis)


def to_path(path):
    return Path(path).expanduser().resolve()


def check_exists(path, is_file=True):
    p = to_path(path)
    if is_file:
        if not p.is_file():
            raise ValueError(f"File does not exist: {path}")
    else:
        if not p.is_dir():
            raise ValueError(f"File does not exist: {path}")


def create_directory(path):
    p = to_path(path)
    dirname = p.parent
    if not dirname.exists():
        os.makedirs(dirname)


def pickle_dump(obj, filename):
    create_directory(filename)
    with open(to_path(filename), "wb") as f:
        pickle.dump(obj, f)


def pickle_load(filename):
    with open(to_path(filename), "rb") as f:
        obj = pickle.load(f)
    return obj


def yaml_dump(obj, filename):
    create_directory(filename)
    with open(to_path(filename), "w") as f:
        yaml.dump(obj, f, default_flow_style=False)


def yaml_load(filename):
    with open(to_path(filename), "r") as f:
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


def seed_torch(seed=35, cudnn_benchmark=False, cudnn_deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    dgl.random.seed(seed)


def save_checkpoints(
    state_dict_objects, misc_objects, is_best, msg=None, filename="checkpoint.pkl"
):
    """
    Save checkpoints for all objects for later recovery.

    Args:
        state_dict_objects (dict): A dictionary of objects to save. The object should
            have state_dict() (e.g. model, optimizer, ...)
        misc_objects (dict): plain python object to save
        filename (str): filename for the checkpoint

    """
    objects = copy.copy(misc_objects)
    for k, obj in state_dict_objects.items():
        objects[k] = obj.state_dict()
    torch.save(objects, filename)
    if is_best:
        shutil.copyfile(filename, "best_checkpoint.pkl")
        if msg is not None:
            logger.info(msg)


def load_checkpoints(state_dict_objects, map_location=None, filename="checkpoint.pkl"):
    """
    Load checkpoints for all objects for later recovery.

    Args:
        state_dict_objects (dict): A dictionary of objects to save. The object should
            have state_dict() (e.g. model, optimizer, ...)
    """
    checkpoints = torch.load(filename, map_location)
    for k, obj in state_dict_objects.items():
        state_dict = checkpoints.pop(k)
        obj.load_state_dict(state_dict)
    return checkpoints


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
