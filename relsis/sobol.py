# -*- coding: utf-8 -*-
import numpy as np
import unittest

__all__ = ["find_sobol_sensitivity"]


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

    for i in xrange(ndim):
        C[:, i] = A[:, i]
        yCi = np.array(map(func, C))
        S1[i] = (np.inner(yA, yCi) / float(nsmp) - f02) / denom
        ST[i] = 1. - (np.inner(yB, yCi) / float(nsmp) - f02) / denom
        C[:, i] = B[:, i]
    return S1, ST


if __name__ == "__main__":
    print "ready to go"