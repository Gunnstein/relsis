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


def _norm(x):
    return np.linalg.norm(x, ord=2)

def get_reliability_index(pf):
    return stats.norm.ppf(1-pf, loc=0., scale=1.)

def truncate_prob_dist(x):
    dtype = x.dtype
    epsneg = np.finfo(dtype).epsneg
    eps = np.finfo(dtype).eps
    x[x<epsneg] = epsneg
    x[x>1.-eps] = 1.-eps

SETTINGS = dict(
    truncate_prob_dist=1e-15)


class RandomVariable:
    """Base class for random variables.

    Interface to any random variable derived from a scipy.stats.rv_continous
    class. Sets the state of the random variable and provides a interface for
    solvers in Relsis.
    """
    def __init__(self, *args, **kwargs):
        self._rv = None

    def ppf(self, probability):
        """Returns the value of x which yields the probabilty P(X<=x)

        The inverse of the cdf also known as the percent point function.
        """
        return self._rv.ppf(probability)

    def cdf(self, x):
        """Returns P(X <= x) of the random variable X.

        The cumulative probability distribution function of the random
        variable.

        """
        return self._rv.cdf(x)

    def to_u(self, x):
        return stats.norm.ppf(self.cdf(x), loc=0., scale=1.)

    def from_u(self, u):
        return self.ppf(stats.norm.cdf(u, loc=0., scale=1.))

    def rvs(self, size=1):
        return self._rv.rvs(size=size)


class NormalRandomVariable(RandomVariable):
    def __init__(self, mean, std):
        """Normally distributed random variable.

        Defined by the mean and standard deviation.

        Arguments
        ---------
        mean, std : float
            Mean and standard deviation of the variable
        """
        self.mean = mean
        self.std = std

        self._rv = stats.norm(loc=mean, scale=std)
        self._kw = {"loc": self.mean, "scale": self.std}

    def __str__(self):
        s = "Normaly distributed random variable, N({loc}, {scale})"
        return s.format(**self._kw)


class UniformRandomVariable(RandomVariable):
    def __init__(self, lower, upper):
        """Uniformly distributed random variable.

        Defined by the lower (included) and upper (excluded) bounds.

        Arguments
        ---------
        mean, std : float
            Mean and standard deviation of the variable
        """
        self.lower = lower
        self.upper = upper

        self._rv = stats.uniform(loc=lower, scale=upper-lower)
        self._kw = {"loc": self.lower, "scale": self.upper-self.lower}

    def __str__(self):
        s = "Uniformly distributed random variable, U({loc}, {loc}+{scale})"
        return s.format(**self._kw)

class TestCase:
    def __init__(self):
        self.a = np.array([78., 12., 0.5, 2., 97., 33])
        self.sampled_trajectories = dict(
                                    trajectory_1=1./3 * np.array([
                                        [0.0, 2.0, 3.0, 0.0, 0.0, 1.0],
                                        [0.0, 2.0, 3.0, 0.0, 0.0, 3.0],
                                        [0.0, 0.0, 3.0, 0.0, 0.0, 3.0],
                                        [2.0, 0.0, 3.0, 0.0, 0.0, 3.0],
                                        [2.0, 0.0, 3.0, 2.0, 0.0, 3.0],
                                        [2.0, 0.0, 1.0, 2.0, 0.0, 3.0],
                                        [2.0, 0.0, 1.0, 2.0, 2.0, 3.0]
                                        ]),
                                    trajectory_2=1./3 * np.array([
                                        [0.0, 1.0, 1.0, 3.0, 3.0, 2.0],
                                        [0.0, 3.0, 1.0, 3.0, 3.0, 2.0],
                                        [2.0, 3.0, 3.0, 3.0, 3.0, 2.0],
                                        [0.0, 3.0, 3.0, 3.0, 3.0, 2.0],
                                        [2.0, 3.0, 3.0, 3.0, 3.0, 0.0],
                                        [2.0, 3.0, 3.0, 3.0, 1.0, 0.0],
                                        [2.0, 3.0, 3.0, 1.0, 1.0, 0.0]
                                        ]),
                                    trajectory_3=1./3 * np.array([
                                        [3.0, 2.0, 0.0, 2.0, 3.0, 0.0],
                                        [3.0, 2.0, 0.0, 0.0, 3.0, 0.0],
                                        [1.0, 2.0, 0.0, 0.0, 3.0, 0.0],
                                        [1.0, 2.0, 0.0, 0.0, 1.0, 0.0],
                                        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                        [1.0, 0.0, 2.0, 0.0, 1.0, 0.0],
                                        [1.0, 0.0, 2.0, 0.0, 1.0, 0.0]
                                        ]),
                                    trajectory_4=1./3 * np.array([
                                        [3.0, 1.0, 2.0, 3.0, 0.0, 1.0],
                                        [3.0, 1.0, 2.0, 3.0, 0.0, 3.0],
                                        [3.0, 1.0, 0.0, 3.0, 0.0, 3.0],
                                        [3.0, 1.0, 0.0, 1.0, 0.0, 3.0],
                                        [3.0, 1.0, 0.0, 1.0, 2.0, 3.0],
                                        [3.0, 3.0, 0.0, 1.0, 2.0, 3.0],
                                        [3.0, 3.0, 0.0, 1.0, 2.0, 3.0]
                                        ]),)
    def __call__(self, x):
        a = self.a
        y = 1.


class SobolTestFunction:
    def __init__(self, a=None):
        """Creates an instance of the sobol test function

        The Sobol test function is defined by

            product_{i=1}^{k} g_i(X_i)
                where g(X_i) = frac{|4X_i-2| + a_i}{1+a_i}

        and is highly nonlinear and nonmonotic. The partial variances and the
        total variance can be determined analytically such that the sensitivity
        indices can be determined analytically.

        Arguments
        ---------
        a : Optional[ndarray]
            The coefficients in the test function. If these are not given, the
            default is set to the one test case given in Saltellis Primer book.


        """
        self.a = a or np.array([78., 12., 0.5, 2., 97., 34.])

    def __call__(self, x):
        """Returns the Sobol analytical

        """
        a = self.a
        return np.product((np.abs(4 * x - 2) + a)/(1 + a))


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
    res = opt.minimize(lambda u: _norm(u), u0,
                       constraints=cons, method='SLSQP')
    alpha = res['jac']
    beta = res['fun']

    if _ls(u0) <= 0:
        beta *= -1.

    result = dict(alpha=alpha, beta=beta)
    return result


def sampler(n_samples, dimensions, corr_matrix=None):
    """Returns (n_samples x dimensions) randomly selected points.

    The function returns a (n_samples x dimensions) sample points which are
    correlated according to the correlation matrx: `corr_matrix`.

    Arguments
    ---------
    n_samples, dimensions : int
        The number of samples and the dimensions

    corr_matrix : Optional[ndarray]
        The correlation matrix defines the linear relationship between the
        variables. Entries should be floats in the range [-1., 1.]. The matrix
        should be positive semidefinite.

    Returns
    -------
    ndarray
        (n_samples x dimensions) array with random sample points.
    """
    if corr_matrix is None:
        x = stats.uniform.rvs(size=(int(n_samples), int(dimensions)),
                              loc=0., scale=1.)
    else:
        Nx = stats.multivariate_normal.rvs(mean=np.zeros(dimensions),
                                           cov=corr_matrix,
                                           size=int(n_samples))
        x = stats.norm.cdf(Nx, loc=0., scale=1)
    return x


def monte_carlo_simulation(limit_state_function, random_variables, n_samples,
                           corr_matrix=None):
    ndim = len(random_variables)
    X0 = sampler(n_samples, ndim, corr_matrix)
    X = np.array([Xi.ppf(X0[:, n])
                 for n, Xi in enumerate(random_variables)]).T
    y = np.array(map(limit_state_function, X))
    return X, y


def sensitivity_analysis2(limit_state_function, random_variables, n_samples,
                         corr_matrix=None):

    ndim = len(random_variables)
    nsmp = int(n_samples / 2)
    X, y, _ = monte_carlo_simulation(limit_state_function, random_variables,
                                     nsmp*2, corr_matrix=corr_matrix)
    yA = y[:nsmp]
    yB = y[nsmp:]
    A = X[:nsmp]
    B = X[nsmp:]

    YC = np.zeros_like(B)

    f02 = yA.mean()**2
    yA2 = np.inner(yA, yA) / float(nsmp)
    denom = yA2 - f02

    C = B.copy()
    S1, ST = np.zeros(ndim), np.zeros(ndim)

    for i in xrange(ndim):
        C[:, i] = A[:, i]
        yCi = np.array(map(limit_state_function, C))
        S1[i] = (np.inner(yA, yCi) / float(nsmp) - f02) / denom
        ST[i] = 1. - (np.inner(yB, yCi) / float(nsmp) - f02) / denom
        C[:, i] = B[:, i]
    return S1, ST, A, B


def sensitivity_analysis(limit_state_function, X, y):
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
        yCi = np.array(map(limit_state_function, C))
        S1[i] = (np.inner(yA, yCi) / float(nsmp) - f02) / denom
        ST[i] = 1. - (np.inner(yB, yCi) / float(nsmp) - f02) / denom
        C[:, i] = B[:, i]
    return S1, ST


def find_morris_trajectory(n_dim, n_lvl, n_jump=None, starting_point=None):
    """Return trajectory using method from original paper of Morris.

    Arguments
    ---------
    n_dim, n_lvl : int
        The number of dimensions and and levels to generate for the grid.
        Note that `n_lvl` must be an even number.

    n_jump : Optional[int]
        The number of levels to jump generating the trajectory. Must be
        smaller than `n_dim`. If `None` the jump parameter is set to
        half the number of levels.

    Returns
    -------
    ndarray
        Morris trajectory, (`n_dim`+1 x `n_dim`) array.

    Raises
    ------
    ValueError
        If the `n_lvl` is not an even number.
    """
    k = int(n_dim)
    p = int(n_lvl)
    if float(p) % 2 != 0.:
        raise ValueError("`n_lvl` must be an even number")
    n_jump = n_jump or float(p) / 2.
    delta = n_jump / float(p-1.)
    if starting_point is None:
        xstar = np.random.randint(low=0, high=p-n_jump, size=k) / float(p-1.)
    else:
        xstar = starting_point/float(p-1)
    B = np.tril(np.ones((k+1, k), np.float), -1)
    J = lambda i, j: np.ones((i, j), np.float)
    D = np.diag(np.random.randint(low=0, high=2, size=k)*2-1).astype(
                                                                np.float)
    P = np.eye(k, dtype=np.float)
    np.random.shuffle(P)
    return np.dot(np.outer(J(k+1, 1), xstar)
        + delta/2.*(np.dot(2.*B-J(k+1, k), D) + J(k+1, k)), P)


def find_trajectories(n_trajec, n_dim, n_lvl, n_jump=None):
    """Return `n_trajec` Morris trajectories.


    Arguments
    ---------
    n_trajec : int
        The number of trajectories to return

    n_dim, n_lvl : int
        The number of dimensions and and levels to generate for the grid.
        Note that `n_lvl` must be an even number.

    n_jump : Optional[int]
        The number of levels to jump generating the trajectory. Must be
        smaller than `n_dim`. If `None` the jump parameter is set to
        half the number of levels.

    Returns
    -------
    ndarray
        Trajectories, (`n_traject` x `n_dim`+1 x `n_dim`) array.

    Raises
    ------
    ValueError
        If the `n_lvl` is not an even number or if
    """
    k = int(n_dim)
    p = int(n_lvl)
    r = int(n_trajec)
    n_cand = r
    candidates = np.zeros((n_cand, k+1, k))
    for i in xrange(n_cand):
        candidates[i] = find_morris_trajectory(k, p, n_jump)

    return candidates

def _q2x(X, random_variables):
    """Transforms the matrix X from [0, 1]-space to the random variable space.

    `X` is a (m x n)-array and `random_variables` is a (1xn)-array.
    """
    return np.array([Xi.ppf(X[:, n])
                 for n, Xi in enumerate(random_variables)]).T

def find_elementary_effects(limit_state_function, random_variables, trajectory):
    k = len(random_variables)

    EE = np.zeros(k, dtype=np.float)
    # X = np.zeros_like(trajectory)
    g = np.zeros(k, dtype=np.float)
    # x = np.zeros(k, dtype=np.float)
    X = _q2x(trajectory, random_variables)
    g = map(limit_state_function, X)

    # dg = trajectory[:-1]-trajectory[1:]
    for l in xrange(k):
        dtj = trajectory[l+1] - trajectory[l]
        n = np.argmax(np.abs(dtj))
        delta = dtj[n]
        EE[n] = (g[l+1] - g[l]) / delta

    return EE





import unittest

class MonteCarloSimulationTestCase(unittest.TestCase):
    def setUp(self):
            ls = lambda x: x[0]**2 + x[1]**2 - 1. / np.pi
            random_variables = [UniformRandomVariable(0., 1.),
                                UniformRandomVariable(0., 1.)]
            self.true = 1.0
            X, y = monte_carlo_simulation(ls, random_variables, 1e6)

            self.estimated = float(y[y<=0].size) / float(y.size) * 4.

    def test_area(self):
        self.assertAlmostEqual(self.true, self.estimated, places=2,
                               msg="Monte Carlo integration failed.")

class FindElementaryEffectsTestCase(unittest.TestCase):
    def setUp(self):
        self.sobol_func = SobolTestFunction()
        self.trajectories = 1. / 3. * np.array([
                                            [
                                                [0.0, 2.0, 3.0, 0.0, 0.0, 1.0],
                                                [0.0, 2.0, 3.0, 0.0, 0.0, 3.0],
                                                [0.0, 0.0, 3.0, 0.0, 0.0, 3.0],
                                                [2.0, 0.0, 3.0, 0.0, 0.0, 3.0],
                                                [2.0, 0.0, 3.0, 2.0, 0.0, 3.0],
                                                [2.0, 0.0, 1.0, 2.0, 0.0, 3.0],
                                                [2.0, 0.0, 1.0, 2.0, 2.0, 3.0]
                                            ],
                                            [
                                                [0.0, 1.0, 1.0, 3.0, 3.0, 2.0],
                                                [0.0, 3.0, 1.0, 3.0, 3.0, 2.0],
                                                [0.0, 3.0, 3.0, 3.0, 3.0, 2.0],
                                                [2.0, 3.0, 3.0, 3.0, 3.0, 2.0],
                                                [2.0, 3.0, 3.0, 3.0, 3.0, 0.0],
                                                [2.0, 3.0, 3.0, 3.0, 1.0, 0.0],
                                                [2.0, 3.0, 3.0, 1.0, 1.0, 0.0]
                                            ],
                                            [
                                                [3.0, 2.0, 0.0, 2.0, 3.0, 0.0],
                                                [3.0, 2.0, 0.0, 0.0, 3.0, 0.0],
                                                [1.0, 2.0, 0.0, 0.0, 3.0, 0.0],
                                                [1.0, 2.0, 0.0, 0.0, 1.0, 0.0],
                                                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                                [1.0, 0.0, 2.0, 0.0, 1.0, 0.0],
                                                [1.0, 0.0, 2.0, 0.0, 1.0, 2.0]
                                            ],
                                            [
                                                [3.0, 1.0, 2.0, 3.0, 0.0, 1.0],
                                                [3.0, 1.0, 2.0, 3.0, 0.0, 3.0],
                                                [3.0, 1.0, 0.0, 3.0, 0.0, 3.0],
                                                [3.0, 1.0, 0.0, 1.0, 0.0, 3.0],
                                                [3.0, 1.0, 0.0, 1.0, 2.0, 3.0],
                                                [3.0, 3.0, 0.0, 1.0, 2.0, 3.0],
                                                [1.0, 3.0, 0.0, 1.0, 2.0, 3.0]
                                            ]
                                            ])
        self.g = np.array([
                    [ 2.194,  2.278,  2.518,  2.476,  1.651,  0.77 ,  0.76 ],
                    [ 1.024,  1.132,  2.425,  2.384,  2.476,  2.443,  1.629],
                    [ 1.519,  2.278,  2.24 ,  2.21 ,  2.443,  1.14 ,  1.098],
                    [ 1.024,  1.063,  2.278,  1.519,  1.498,  1.656,  1.629]])

    def test_sobol_func(self):
        tjs = self.trajectories
        tf = self.sobol_func
        g = np.array(
                [np.round(tf(x), 3)  for tj in tjs for x in tj]).reshape(tjs.shape[:2])

        np.testing.assert_allclose(g, self.g, rtol=1e-3,
                                   err_msg="Sobol test function not ok")

    def test_find_morris_trajectory(self):
        true = np.array([2. / 3.]*6)
        tjs = find_morris_trajectory(6, 4, 2)
        deltas = np.abs(tjs[1:] - tjs[:-1])
        np.testing.assert_allclose(np.sum(deltas, axis=0), true)
        np.testing.assert_allclose(np.sum(deltas, axis=1), true)











if __name__ == '__main__':
    # unittest.main()

    tjs = 1. / 3. * np.array([
                                            [
                                                [0.0, 2.0, 3.0, 0.0, 0.0, 1.0],
                                                [0.0, 2.0, 3.0, 0.0, 0.0, 3.0],
                                                [0.0, 0.0, 3.0, 0.0, 0.0, 3.0],
                                                [2.0, 0.0, 3.0, 0.0, 0.0, 3.0],
                                                [2.0, 0.0, 3.0, 2.0, 0.0, 3.0],
                                                [2.0, 0.0, 1.0, 2.0, 0.0, 3.0],
                                                [2.0, 0.0, 1.0, 2.0, 2.0, 3.0]
                                            ],
                                            [
                                                [0.0, 1.0, 1.0, 3.0, 3.0, 2.0],
                                                [0.0, 3.0, 1.0, 3.0, 3.0, 2.0],
                                                [0.0, 3.0, 3.0, 3.0, 3.0, 2.0],
                                                [2.0, 3.0, 3.0, 3.0, 3.0, 2.0],
                                                [2.0, 3.0, 3.0, 3.0, 3.0, 0.0],
                                                [2.0, 3.0, 3.0, 3.0, 1.0, 0.0],
                                                [2.0, 3.0, 3.0, 1.0, 1.0, 0.0]
                                            ],
                                            [
                                                [3.0, 2.0, 0.0, 2.0, 3.0, 0.0],
                                                [3.0, 2.0, 0.0, 0.0, 3.0, 0.0],
                                                [1.0, 2.0, 0.0, 0.0, 3.0, 0.0],
                                                [1.0, 2.0, 0.0, 0.0, 1.0, 0.0],
                                                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                                [1.0, 0.0, 2.0, 0.0, 1.0, 0.0],
                                                [1.0, 0.0, 2.0, 0.0, 1.0, 2.0]
                                            ],
                                            [
                                                [3.0, 1.0, 2.0, 3.0, 0.0, 1.0],
                                                [3.0, 1.0, 2.0, 3.0, 0.0, 3.0],
                                                [3.0, 1.0, 0.0, 3.0, 0.0, 3.0],
                                                [3.0, 1.0, 0.0, 1.0, 0.0, 3.0],
                                                [3.0, 1.0, 0.0, 1.0, 2.0, 3.0],
                                                [3.0, 3.0, 0.0, 1.0, 2.0, 3.0],
                                                [1.0, 3.0, 0.0, 1.0, 2.0, 3.0]
                                            ]
                                            ])

    tf = SobolTestFunction()
    random_variables = [UniformRandomVariable(0., 1.)] * 6


    EE = np.array(
        map(lambda tj: find_elementary_effects(tf, random_variables, tj), tjs))
    print EE
    for mstar, m, s in zip(
        np.mean(np.abs(EE), axis=0), np.mean(EE, axis=0), np.std(EE, axis=0)):
        print " | ".join(map("{0:>6.3f}".format, (mstar, m, s)))
















