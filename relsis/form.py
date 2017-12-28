# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize
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
    alpha = res['jac'] / np.linalg.norm(res['jac'], 2)
    beta = res['fun']

    if _ls(u0) <= 0:
        beta *= -1.

    result = dict(alpha=alpha, beta=beta)
    return result
