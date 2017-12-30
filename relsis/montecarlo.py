# -*- coding: utf-8 -*-
import numpy as np
import unittest
from randomvariables import UniformRandomVariable
import multiprocessing
import sampling
import utils

__all__ = ["monte_carlo_simulation"]


def monte_carlo_simulation(func, random_variables, n_smp, corr_matrix=None,
                           sampling_method='crude', n_cpu=1):
    """Perform a MC simulation on function with random variables

    This function evaluates `func` by sampling the random variables `n_smp`
    times and returns the design matrix and the function evaluations.

    Arguments
    ---------
    func : function
        The function should take an array of variables corresponding to
        realizations of the random variables given in the array
        `random_variables`. It must be possible to pickle the function, e.g
        it is not possible to use class methods or lambda functions.

    random_variables : array
        An array of RandomVariable instances.

    n_smp : int
        The number of function evaluations to perform in the simulation

    corr_matrix : Optional[ndarray]
        An square array with len(`random_variables`) dimensions defining the
        correlation between the different random variables. Only sampling
        method available is `crude`.

    sampling method : str
        Sample method, possible values: `crude`, `sobol`, `latin_random`,
                                        `latin_center`, `latin_edge`.

        If `corr_matrix` is given, the sampling method overridden to `crude`.

    n_cpu : Optional[int]
        The number of cpu's to use in the simulation.

    Returns
    -------
    X : ndarray
        The design matrix containing the realizations of the random variables
        for each of the function evaluations.

    y : ndarray
        The return values from the function evaluations.
    """
    n_dim = len(random_variables)
    n_strata = np.int(np.round(n_smp**(1./n_dim), 0))
    if corr_matrix is not None:
        sampling_method = 'crude'
    if sampling_method == 'crude':
        X0 = sampling.get_sample_crude(n_smp, n_dim, corr_matrix)
    elif sampling_method == 'sobol':
        X0 = sampling.get_sample_sobol(n_smp, n_dim)
    elif sampling_method == 'latin_random':
        X0 = sampling.get_sample_latin_random(n_strata, n_dim)
    elif sampling_method == 'latin_center':
        X0 = sampling.get_sample_latin_center(n_strata, n_dim)
    elif sampling_method == 'latin_edge':
        X0 = sampling.get_sample_latin_edge(n_strata, n_dim)
    else:
        raise ValueError("Sampling method not recognized.")
    X = np.array([Xi.ppf(X0[:, n])
                 for n, Xi in enumerate(random_variables)]).T
    pool = multiprocessing.Pool(n_cpu)
    y = np.asfarray(pool.map(func, X))
    return X, y


# def _circle(x):
#     return x[0]**2 + x[1]**2 - 1. / np.pi


# def _triangle(x):
#     return x[0] - x[1]


# class MonteCarloSimulationTestCase(unittest.TestCase):
#     def area_circle(self, sampling_method):
#         random_variables = [UniformRandomVariable(0., 1.),
#                             UniformRandomVariable(0., 1.)]
#         true = 1.0
#         X, y = monte_carlo_simulation(_circle, random_variables, 1e6,
#                                       sampling_method=sampling_method)

#         estimated = float(y[y<=0].size) / float(y.size) * 4.
#         print estimated
#         self.assertAlmostEqual(true, estimated, places=2,
#                                msg="Monte Carlo integration failed.")

#     def test_area_circle_crude(self):
#         self.area_circle('crude')


# if __name__ == "__main__":
    # unittest.main()
    # random_variables = [UniformRandomVariable(0., 1.),
    #                     UniformRandomVariable(0., 1.)]
    # X, y = monte_carlo_simulation(_circle, random_variables, 20000, n_cpu=4,
    #                               sampling_method='sobol')
    # y1 = map(_circle, X)

    # p = multiprocessing.Pool(1)
    # y2 = np.asfarray(p.map(_circle, X))
    # print type(y1)
    # print type(y2)
    # np.testing.assert_equal(y1, y2)



