# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np

__all__ = ['get_sample_latin_random', 'get_sample_latin_center',
           'get_sample_latin_edge']

def get_sample_latin_random(n_strata, n_dim):
    """Lating hypercube sampling with random selection in each cube.

    The number of samples is (n_smp = n_strata ** n_dim)

    """
    dtype = np.float
    X = np.mgrid[[slice(n_strata)]*n_dim].T.reshape(-1, n_dim).astype(dtype)
    X += np.random.uniform(0.0, 1.0,
                           size=(n_strata**n_dim, n_dim)).astype(dtype)
    X /= np.float(n_strata)
    return X


def get_sample_latin_center(n_strata, n_dim):
    """Lating hypercube sampling with selection at the center in each cube.

    The number of samples is (n_smp = n_strata ** n_dim)

    """
    dtype = np.float
    X = np.mgrid[[slice(n_strata)]*n_dim].T.reshape(-1, n_dim).astype(dtype)
    X += .5
    X /= np.float(n_strata)
    return X


def get_sample_latin_edge(n_strata, n_dim):
    """Lating hypercube sampling with selection at edge(corner) of each cube.

    The number of samples is (n_smp = n_strata ** n_dim)

    """
    dtype = np.float
    X = np.mgrid[[slice(n_strata)]*n_dim].T.reshape(-1, n_dim).astype(dtype)
    X /= np.float(n_strata-1)
    return X


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import itertools

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    cols = ['k','r', 'b', 'g']
    for n, method_name in enumerate(__all__):
        if "random" in method_name:
            continue
        method = eval(method_name)
        X = method(3, 3)
        print(X.shape)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=cols[n])

    fig.tight_layout()
    plt.show(block=True)