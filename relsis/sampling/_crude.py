# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import scipy.stats as stats


def get_sample_crude(n_smp, n_dim, corr_matrix=None):
    """Returns (n_smp x n_dim) randomly selected points.

    The function returns a (n_smp x n_dim) sample points which are
    correlated according to the correlation matrx: `corr_matrix`.

    Arguments
    ---------
    n_smp, n_dim : int
        The number of samples and the dimensions (number of random variables).

    corr_matrix : Optional[ndarray]
        The correlation matrix defines the linear relationship between the
        variables. Entries should be floats in the range [-1., 1.]. The matrix
        should be positive semidefinite.

    Returns
    -------
    ndarray
        (n_smp x n_dim) array with random sample points.
    """
    if corr_matrix is None:
        x = stats.uniform.rvs(size=(int(n_smp), int(n_dim)),
                              loc=0., scale=1.)
    else:
        Nx = stats.multivariate_normal.rvs(mean=np.zeros(n_dim),
                                           cov=corr_matrix,
                                           size=int(n_smp))
        x = stats.norm.cdf(Nx, loc=0., scale=1)
    return x
