# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import scipy
import unittest
from .. import *


__all__ = ["find_sobol_sensitivity"]


def find_sensitivity_sobol(func, X, y, n_resample=500, alpha=5.0, n_cpu=1):
    """Determine the first (S1) and total (ST) Sobol sensitivity indices.

    The estimators recommended in [Saltelli2010] are used for the first and
    total order sensitivities. Note that the sampling strategy suggested
    in Saltelli is not implemented in the function, specifically the function
    expects an (2Nxd) array and splits that into the sample and resample
    matrix by randomly selecting N rows from X, y. Saltellis suggestion is to
    generate a (Nx2d) matrix, where each row is spaced over [0, 1], the
    sample/resample matrix is then the half of the columns for each of them.

    Empirical bootstrapping is used to determine the confidence intervals.

    Arguments
    ---------
    func : function
        The function should take an array of variables corresponding to
        realizations of the random variables given in the array
        `random_variables`. It must be possible to pickle the function, e.g
        it is not possible to use class methods or lambda functions if n_cpu>1.

    X : ndarray
        The design matrix with realizations of random variables in the columns.

    y : ndarray
        Output of the limit state function corresponding to input from the
        design matrix.

    n_resample : Optional[int]
        The number of bootstrap resamples for confidence interval estimate.
        If n_resample <= 0, bootstrap will not be performed and the the
        function returns None for S1conf and STconf

    alpha : Optional[float]
        The significance level (in percent) for the confidence interval.

    n_cpu : Optional[int or 'str']
        The number of cpu's to use in the simulation. If 'max' is specified,
        the maximum number of cpus is used.

    Returns
    -------
    S1, ST, S1conf, STconf: array
        The first order and total Sobol sensitivity indices and the estimates
        for the confidence intervals.
        If n_resample <= 0, will not be performed and the the function returns
        None for S1conf and STconf
    """
    if n_resample is None:
        n_resample = -1
    ntot, ndim = X.shape
    if X.shape[0] % 2 != 0:
        Warning("The number of samples is not even, dropping one point!")
        ntot -= 1
    n = np.arange(0, ntot).astype(np.int)
    np.random.shuffle(n)
    nsmp = int(ntot / 2)

    yA = y[n[:nsmp]]
    yB = y[n[nsmp:]]
    A = X[n[:nsmp]]
    B = X[n[nsmp:]]

    var_y = np.var(y[:ntot])
    AB = A.copy()
    S1 = np.zeros(ndim, dtype=np.float)
    ST = np.zeros_like(S1)

    bootstrap = n_resample > 0
    if bootstrap:
        S1conf = np.zeros((ndim, 2), dtype=np.float)
        STconf = np.zeros_like(S1conf)
        S1s = np.zeros(n_resample, dtype=np.float)
        STs = np.zeros_like(S1s)
        nl = np.round(alpha/200. * n_resample).astype(np.int)
        nu = np.round(1. - alpha/200. * n_resample).astype(np.int)
    else:
        S1conf, STconf = None, None

    for i in range(ndim):
        AB[:, i] = B[:, i]
        yAB = utils._map(func, AB, n_cpu)
        S1[i], ST[i] = _first_and_totalorder_sensitivity_indices(
            yA, yAB, yB, var_y=var_y)
        AB[:, i] = A[:, i]

        if bootstrap:
            for j in range(n_resample):
                n = np.random.choice(nsmp, size=nsmp)
                S1sj, STsj = _first_and_totalorder_sensitivity_indices(
                    np.take(yA, n), np.take(yAB, n), np.take(yB, n))
                S1s[j], STs[j] = S1sj - S1[i], STsj - ST[i]

            S1s, STs = np.sort(S1s), np.sort(STs)
            S1conf[i, :] = np.array([S1s[nl], S1s[nu]])
            STconf[i, :] = np.array([STs[nl], STs[nu]])

    return S1, ST, S1conf, STconf


def _first_and_totalorder_sensitivity_indices(yA, yAB, yB, var_y=None):
    """Returns the first and total order sensitivity indices.

    """
    var_y = var_y or np.var(np.concatenate((yA, yB)))
    yAB_A = yAB - yA
    V_Ei = np.mean(yB*yAB_A)
    E_Vi = 1. / 2. * np.mean(yAB_A ** 2.)
    return V_Ei / var_y, E_Vi / var_y
