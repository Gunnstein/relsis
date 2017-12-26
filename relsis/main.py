# -*- coding: utf-8 -*-
"""

TODO:
=====

Sampler
-------
- The sampler must support correlated variables.
- The sampler must support crude/random sampling.
- The sampler must support sampling of Morris trajectories.

- The sampler should support a quasirandom sampling method, e.g BOSLHS, Sobol.

Solver
------
- The package must provide a Monte Carlo solver function.

- The MC solver should allow multiple CPU's to be used.
- The MC solver should save the results to a file as we go.


Sensitivity analysis
--------------------
- The package must support variance based sensitivity analysis.
- The package must support elementary effects (EE) sensitivity analysis.


Testing
-------
- Implement the reference case in 3.6 in Saltelli's primer book.


The s
- The sampler should support crude s
- Write unit tests.
- Sensitivity analysis, alpha factors at design point, global `Saltelli` indices.
- Implement different sampling strategies, LHS, BOSLHS, Sobol sequence,
  Hansol.
-

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt
import scipy.misc as misc
import itertools
from randomvariables import *
from utils import *


def find_sobol_sensitivity(func, X, y):
    """Determine the first (S1) and total (ST) Sobol sensitivity indices.

    Arguments
    ---------
    func : function
        The function should return a scalar value and take a array of variables
        corresponding to realizations of the random variables given in the
        array `random_variables`.

    X : ndarray
        The design matrix with realizations of random variables in the columns.

    y : ndarray
        Output of the limit state function corresponding to input from the
        design matrix.
    """
    ntot, ndim = X.shape
    if X.shape[0] % 2 != 0:
        Warning("The number of samples is not even, dropping one point!")
        ntot -= 1
    nsmp = ntot / 2

    yA = y[:nsmp]
    yB = y[nsmp:ntot]
    A = X[:nsmp]
    B = X[nsmp:ntot]

    YC = np.zeros_like(B)

    f02 = yA.mean()**2
    yA2 = np.inner(yA, yA) / float(nsmp)
    denom = yA2 - f02

    C = B.copy()
    S1, ST = np.zeros(ndim), np.zeros(ndim)

    for i in xrange(ndim):
        C[:, i] = A[:, i]
        yCi = np.array(map(func, C))
        S1[i] = (np.inner(yA, yCi) / float(nsmp) - f02) / denom
        ST[i] = 1. - (np.inner(yB, yCi) / float(nsmp) - f02) / denom
        C[:, i] = B[:, i]
    return S1, ST


if __name__ == '__main__':
    # unittest.main()
    pass
