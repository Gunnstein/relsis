# -*- coding: utf-8 -*-
import numpy as np
import unittest
from randomvariables import UniformRandomVariable
from sampling import crude_sampler

__all__ = ["monte_carlo_simulation"]

def monte_carlo_simulation(func, random_variables, n_smp, corr_matrix=None):
    """Perform a MC simulation on function with random variables

    This function evaluates `func` by sampling the random variables `n_smp`
    times and returns the design matrix and the function evaluations.

    Arguments
    ---------
    func : function
        The function should return a scalar value and take a array of variables
        corresponding to realizations of the random variables given in the
        array `random_variables`.

    random_variables : array
        An array of RandomVariable instances.

    n_smp : int
        The number of function evaluations to perform in the simulation

    corr_matrix : ndarray
        An square array with len(`random_variables`) dimensions defining the
        correlation between the different random variables.

    Returns
    -------
    X : ndarray
        The design matrix containing the realizations of the random variables
        for each of the function evaluations.

    y : ndarray
        The return values from the function evaluations.
    """
    n_dim = len(random_variables)
    X0 = crude_sampler(n_smp, n_dim, corr_matrix)
    X = np.array([Xi.ppf(X0[:, n])
                 for n, Xi in enumerate(random_variables)]).T
    y = np.array(map(func, X))
    return X, y

class MonteCarloSimulationTestCase(unittest.TestCase):
    def test_area_circle(self):
        ls = lambda x: x[0]**2 + x[1]**2 - 1. / np.pi
        random_variables = [UniformRandomVariable(0., 1.),
                            UniformRandomVariable(0., 1.)]
        true = 1.0
        X, y = monte_carlo_simulation(ls, random_variables, 1e6)

        estimated = float(y[y<=0].size) / float(y.size) * 4.
        self.assertAlmostEqual(true, estimated, places=2,
                               msg="Monte Carlo integration failed.")

    def test_area_triangle(self):
        ls = lambda x: x[0] - x[1]
        random_variables = [UniformRandomVariable(0., np.sqrt(2)),
                            UniformRandomVariable(0., np.sqrt(2))]
        true = 1.
        X, y = monte_carlo_simulation(ls, random_variables, 1e6)

        estimated = float(y[y<=0].size) / float(y.size) * 2
        self.assertAlmostEqual(true, estimated, places=2,
                               msg="Monte Carlo integration failed.")


if __name__ == "__main__":
    unittest.main()

