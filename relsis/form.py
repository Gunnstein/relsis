# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize
import unittest
import randomvariables


def form_solver(limit_state_function, random_variables):
    """
    Evaluates the integral `limit_state_function(random_variables) <= 0` by
    first order reliability method. Transforms the limit state
    function to the standard normal space and determines the integral by
    finding the most probable point with the following minimization problem:

        minimize ||u||_2 subject to g(u) = 0

    Arguments
    ---------
    limit_state_function : function
        The function should return a scalar value and take a array of variables
        corresponding to realizations of the random variables given in the
        array `random_variables`.

    random_variables : array
        An array of RandomVariable instances.

    Returns
    -------
        dict
            A dictionary with solution.
    """
    u0 = np.zeros(len(random_variables))

    def _ls(u):
        """Transforms: u -> x space and returns limit_state_function.

        """
        x = np.array([Xi.from_u(ui) for Xi, ui in zip(random_variables, u)])
        return limit_state_function(x)


    cons = dict(type='eq', fun=_ls)
    res = scipy.optimize.minimize(lambda u: np.linalg.norm(u, 2), u0,
                                  constraints=cons, method='SLSQP')
    alpha = res['jac']
    beta = res['fun']

    if _ls(u0) <= 0:
        beta *= -1.

    result = dict(alpha=alpha, beta=beta)
    return result


class TestFormSolver(unittest.TestCase):
    def setUp(self):
        self.random_variables = [randomvariables.NormalRandomVariable(20., 4.),
                                 randomvariables.NormalRandomVariable(5., 3.)]
        self.limit_state_function = lambda x: x[0] - x[1]
        self.beta_true = (20.-5.) / np.sqrt(4**2+3**2) # beta = 3

    def test_form_solver(self):
        result = form_solver(self.limit_state_function, self.random_variables)
        beta = result['beta']
        self.assertAlmostEqual(beta, self.beta_true)



if __name__ == "__main__":
    unittest.main()