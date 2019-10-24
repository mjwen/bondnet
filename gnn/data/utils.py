import os
import pickle


def create_directory(filename):
    filename = os.path.abspath(filename)
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    return filename


def pickle_dump(obj, filename):
    create_directory(filename)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj
