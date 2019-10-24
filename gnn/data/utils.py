import os
import pickle


def create_directory(filename):
    filename = os.path.abspath(filename)
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    return filename


def print_dict(d, indent=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print(' ' * indent + str(k) + ':')
            print_dict(v, indent + 4)
        else:
            print(' ' * indent + '{0}: {1}'.format(k, v))


def pickle_dump(obj, filename):
    create_directory(filename)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj
