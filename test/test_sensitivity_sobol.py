# -*- coding: utf-8 -*-
if __package__ is None:
    import sys
    sys.path.append('../')
import unittest
from relsis import *
import numpy as np



class TestFindSobolSensitivity(unittest.TestCase):
    def setUp(self):
        self.rvs = [NormalRandomVariable(0., 1.),
                    NormalRandomVariable(0., 2.),
                    NormalRandomVariable(0., 3.),
                    NormalRandomVariable(0., 4.),]

        self.S_linear_true = np.array([0.03, 0.13, 0.30, 0.53])
        self.X, self.y = monte_carlo_simulation(self.func, self.rvs, 1e5,
                                                sampling_method='latin_center')

    def func(self, x):
        return 2. * x[0] + 2.*x[1] + 2.*x[2] + 2.*x[3]

    def test_linear(self):
        S1, ST = find_sensitivity_sobol(self.func, self.X, self.y)
        np.testing.assert_almost_equal(ST, self.S_linear_true, 2)
        np.testing.assert_almost_equal(S1, self.S_linear_true, 2)
        self.assertAlmostEqual(np.sum(S1), 1.0, places=1)


if __name__ == '__main__':
    unittest.main()