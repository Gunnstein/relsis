# -*- coding: utf-8 -*-
import numpy as np
import multiprocessing
import scipy
if __package__ is None:
    import sys
    sys.path.append('../..')
import unittest
from relsis import *


__all__ = ["find_sobol_sensitivity"]


def find_sensitivity_sobol(func, X, y, n_resample=1000, conf_lvl=.95,
                           n_cpu=1):
    """Determine the first (S1) and total (ST) Sobol sensitivity indices.

    The estimators recommended in [Saltelli2010] are used for the first and
    total order sensitivities. Bootstrapping is used to determine the
    confidence intervals.

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

    n_resample : int
        The number of time to bootstrapa resamples for confidence interval
        estimate.

    conf_lvl : float
        The confidence level for the confidence interval.

    n_cpu : Optional[int]
        The number of cpu's to use in the simulation.

    Returns
    -------
    S1, ST, S1conf, STconf: array
        The first order and total Sobol sensitivity indices and the estimates
        for the confidence intervals.
    """
    ntot, ndim = X.shape
    if X.shape[0] % 2 != 0:
        Warning("The number of samples is not even, dropping one point!")
        ntot -= 1
    nsmp = ntot / 2
    N = np.float(nsmp)


    yA = y[:nsmp]
    yB = y[nsmp:ntot]
    A = X[:nsmp]
    B = X[nsmp:ntot]

    var_y = np.var(y[:ntot])

    AB = A.copy()
    pool = multiprocessing.Pool(n_cpu)

    S1 = np.zeros(ndim, dtype=np.float)
    S1conf = np.zeros_like(S1)
    ST = np.zeros_like(S1)
    STconf = np.zeros_like(S1)


    S1s = np.zeros(n_resample, np.float)
    STs = np.zeros_like(S1s)
    n_bootstrap = np.int(nsmp / (ndim + 2))
    z = scipy.stats.norm.ppf(0.5 + conf_lvl / 2.)

    for i in xrange(ndim):
        AB[:, i] = B[:, i]
        yAB = np.array(pool.map(func, AB))
        S1[i], ST[i] = _first_and_totalorder_sensitivity_indices(
            yA, yAB, yB, var_y=var_y)
        AB[:, i] = A[:, i]

        for j in xrange(n_resample):
            n = np.random.choice(nsmp, size=n_bootstrap)
            S1s[j], STs[j] = _first_and_totalorder_sensitivity_indices(
                yA[n], yAB[n], yB[n])

        S1conf[i] = z * S1s.std()
        STconf[i] = z * S1s.std()

    pool.close()
    pool.join()
    return S1, ST, S1conf, STconf


def _first_and_totalorder_sensitivity_indices(yA, yAB, yB, var_y=None):
    """Returns the first and total order sensitivity indices.

    """
    var_y = var_y or np.var(np.concatenate((yA, yB)))
    yAB_A = yAB - yA
    V_Ei = np.mean(yB*yAB_A)
    E_Vi = 1. / 2. * np.mean(yAB_A ** 2.)
    return V_Ei / var_y, E_Vi / var_y
