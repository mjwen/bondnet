from gnn.utils import expand_path


def write_error(errors, ids, sort=True, filename="error.txt"):
    """
    Write the error to file.

    Args:
        errors (list): errors.
        ids (list): ids associated with errors.
        sort (bool): whether to sort the error from low to high.
        filename (str): filename to write out the result.
    """
    if sort:
        errors, ids = zip(*sorted(zip(errors, ids), key=lambda pair: pair[0]))
    with open(expand_path(filename), "w") as f:
        f.write("# error    id\n")
        for e, i in zip(errors, ids):
            f.write("{:13.5e}    {}\n".format(e, i))
