# -*- coding: utf-8 -*-
if __package__ is None:
    import sys
    sys.path.append('../')
import unittest
from relsis import *
from relsis.sensitivity._morris import (find_morris_trajectory,
                                        find_elementary_effects)
import numpy as np


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


if __name__ == '__main__':
    unittest.main()