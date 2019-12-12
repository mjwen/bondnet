import os
import time
import pickle
import yaml
import random
import numpy as np
import torch
import dgl


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
    create_directory(filename)
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj


def yaml_dump(obj, filename):
    filename = expand_path(filename)
    create_directory(filename)
    with open(filename, "w") as f:
        yaml.dump(obj, f, default_flow_style=False)


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
