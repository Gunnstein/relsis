# -*- coding: utf-8 -*-
import numpy as np
import multiprocessing
if __package__ is None:
    import sys
    sys.path.append('../..')
import unittest
from relsis import *


__all__ = ["find_sobol_sensitivity"]


def find_sensitivity_sobol(func, X, y, n_cpu=1):
    """Determine the first (S1) and total (ST) Sobol sensitivity indices.


    Arguments
    ---------
    func : function
        The function should take an array of variables corresponding to
        realizations of the random variables given in the array
        `random_variables`. It must be possible to pickle the function, e.g
        it is not possible to use class methods or lambda functions.

    X : ndarray
        The design matrix with realizations of random variables in the columns.

    y : ndarray
        Output of the limit state function corresponding to input from the
        design matrix.

    n_cpu : Optional[int]
        The number of cpu's to use in the simulation.

    Returns
    -------
    S1, ST : array
        The first order and total Sobol sensitivity indices.
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
    pool = multiprocessing.Pool(n_cpu)
    for i in xrange(ndim):
        C[:, i] = A[:, i]
        yCi = np.array(pool.map(func, C))
        S1[i] = (np.inner(yA, yCi) / float(nsmp) - f02) / denom
        ST[i] = 1. - (np.inner(yB, yCi) / float(nsmp) - f02) / denom
        C[:, i] = B[:, i]
    pool.close()
    pool.join()
    return S1, ST
