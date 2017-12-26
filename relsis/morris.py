# -*- coding: utf-8 -*-
import numpy as np
import unittest
import randomvariables
import utils

__all__ = ["find_sensitivity_morris"]


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
        If the `n_lvl` is not an even number.
    """
    k = int(n_dim)
    p = int(n_lvl)
    r = int(n_trajec)
    n_cand = r
    candidates = np.zeros((n_cand, k+1, k))
    for i in xrange(n_cand):
        candidates[i] = find_morris_trajectory(k, p, n_jump)

    return candidates

def find_elementary_effects(func, random_variables, trajectory):
    """Return the elementary effects for the trajectory

    """
    k = len(random_variables)
    EE = np.zeros(k, dtype=np.float)
    g = np.zeros(k, dtype=np.float)
    X = utils.q2x(trajectory, random_variables)
    g = map(func, X)
    for l in xrange(k):
        dtj = trajectory[l+1] - trajectory[l]
        n = np.argmax(np.abs(dtj))
        delta = dtj[n]
        EE[n] = (g[l+1] - g[l]) / delta

    return EE


def find_sensitivity_morris(func, random_variables, n_trajec, n_lvl, n_jump=None):
    """Returns the sensitivity measures by the method of Morris.

    The mean, the mean of the absolutt value and the standard deviation
    of the elementary effects serves as sensitivity indices. The measures
    are obtained from a series of trajectories in the random variable space

    Arguments
    ---------
    func : function
        The function should return a scalar value and take a array of variables
        corresponding to realizations of the random variables given in the
        array `random_variables`.

    random_variables : array
        An array of RandomVariable instances.

    n_trajec : int
        The number of trajectories to calculate elementary effects for.

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
        If the `n_lvl` is not an even number.
    """
    k = len(random_variables)
    trajectories = find_trajectories(n_trajec, k, n_lvl, n_jump)
    utils.truncate_prob_dist(trajectories)
    EE = np.array(
        map(lambda trajectory: find_elementary_effects(
                                               func,
                                               random_variables,
                                               trajectory), trajectories))
    result = {
        "mu_star": np.mean(np.abs(EE), axis=0),
        "mu": np.mean(EE, axis=0),
        "sigma": np.std(EE, axis=0),
    }
    return result



class TestMorris(unittest.TestCase):
    def setUp(self):
        self.func = utils.SobolTestFunction()
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
        self.elementary_effects = np.array([
                            [-0.063, -0.360,  1.321, -1.238, -0.016,  0.127],
                            [-0.061,  0.162,  1.940,  1.221,  0.050, -0.138],
                            [ 0.057, -0.349, -1.954, -1.139,  0.045, -0.063],
                            [ 0.041,  0.237, -1.823,  1.139, -0.031,  0.059]])
        self.g = np.array([
                    [ 2.194,  2.278,  2.518,  2.476,  1.651,  0.77 ,  0.76 ],
                    [ 1.024,  1.132,  2.425,  2.384,  2.476,  2.443,  1.629],
                    [ 1.519,  2.278,  2.24 ,  2.21 ,  2.443,  1.14 ,  1.098],
                    [ 1.024,  1.063,  2.278,  1.519,  1.498,  1.656,  1.629]])

    def test_func(self):
        tjs = self.trajectories
        tf = self.func
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

    def test_find_elementary_effects(self):
        true = self.elementary_effects
        trjs = self.trajectories
        EE = []
        ls = self.func
        rv = [randomvariables.UniformRandomVariable(0., 1.)] * 6
        for trj in trjs:
            EE.append(find_elementary_effects(self.func, rv, trj))
        EE = np.round(EE, 3)
        np.testing.assert_allclose(np.array(EE), true, rtol=1e-3)

if __name__ == "__main__":
    unittest.main()