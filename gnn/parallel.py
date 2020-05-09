import multiprocessing as mp
import random
import numpy as np


def parmap2(f, X, *args, tuple_X=False, nprocs=mp.cpu_count()):
    r"""Parallelism over data.

    This is to mimic ``multiprocessing.Pool.map``, which requires the function ``f`` to be
    picklable. This function does not have this restriction and allows extra arguments to
    be used for the function ``f``.

    Parameters
    ----------
    f: function
        The function that operates on the data.

    X: list
        Data to be parallelized.

    args: args
        Extra positional arguments needed by the function ``f``.

    tuple_X: bool
        This depends on ``X``. It should be set to ``True`` if multiple arguments are
        parallelized and set to ``False`` if only one argument is parallelized. See
        ``Example`` below.

    nprocs: int
        Number of processors to use.


    Return
    ------
    list
        A list of results, corresponding to ``X``.

    Note
    ----
    Although the arguments of the function does not need to be pickable, the returned
    value needs to.

    This function is implemented using ``multiprocessing.Pipe``. The data is subdivided
    into ``nprocs`` groups and then each group of data is distributed to a process. The
    results from each group are then assembled together.  The data is shuffled to balance
    the load in each process.  See :meth:`kliff.parallel.parmap1` for another
    implementation that uses ``multiprocessing.Queue``.

    Example
    -------
    >>> def func(x, y, z=1):
    >>>     return x+y+z
    >>> X = range(3)
    >>> Y = range(3)
    >>> parmap2(func, X, 1, nprocs=2)  # [2,3,4]
    >>> parmap2(func, X, 1, 1, nprocs=2)  # [2,3,4]
    >>> parmap2(func, zip(X, Y), tuple_X=True, nprocs=2)  # [1,3,5]
    >>> parmap2(func, zip(X, Y), 1, tuple_X=True, nprocs=2)  # [1,3,5]
    """

    # shuffle and divide into nprocs equally-numbered parts
    if tuple_X:
        pairs = [(i, *x) for i, x in enumerate(X)]
    else:
        pairs = [(i, x) for i, x in enumerate(X)]
    random.shuffle(pairs)
    groups = np.array_split(pairs, nprocs)

    processes = []
    managers = []
    for i in range(nprocs):
        manager_end, worker_end = mp.Pipe(duplex=False)
        p = mp.Process(target=_func2, args=(f, groups[i], args, worker_end))
        p.daemon = True
        p.start()
        processes.append(p)
        managers.append(manager_end)
    results = []
    for m in managers:
        results.extend(m.recv())
    for p in processes:
        p.join()

    return [r for i, r in sorted(results)]


def _func2(f, iX, args, worker_end):
    results = []
    for ix in iX:
        i = ix[0]
        x = ix[1:]
        results.append((i, f(*x, *args)))
    worker_end.send(results)
