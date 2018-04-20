# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from relsis import *
import unittest
import numpy as np


def func(x):
    return 2. * x[0] + 2.*x[1] + 2.*x[2] + 2.*x[3]


class TestFindSensitivitySobol(unittest.TestCase):
    def setUp(self):
        self.rvs = [NormalRandomVariable(0., 1.),
                    NormalRandomVariable(0., 2.),
                    NormalRandomVariable(0., 3.),
                    NormalRandomVariable(0., 4.),]

        self.S_linear_true = np.array([0.03, 0.13, 0.30, 0.53])
        self.X, self.y = monte_carlo_simulation(func, self.rvs, 1e5,
                                                sampling_method='latin_center')

    def test_linear(self):
        S1, ST, __, __ = find_sensitivity_sobol(func, self.X, self.y,
                                                n_resample=None)
        np.testing.assert_almost_equal(ST, self.S_linear_true, 2)
        np.testing.assert_almost_equal(S1, self.S_linear_true, 2)
        self.assertAlmostEqual(np.sum(S1), 1.0, places=1)


if __name__ == '__main__':
    unittest.main()