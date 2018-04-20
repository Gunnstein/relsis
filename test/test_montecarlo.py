# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from relsis import *
import unittest
import numpy as np

def _circle(x):
        return x[0]**2 + x[1]**2 - 1. / np.pi

class MonteCarloSimulationTestCase(unittest.TestCase):

    def area_circle(self, sampling_method):
        random_variables = [UniformRandomVariable(0., 1.),
                            UniformRandomVariable(0., 1.)]
        true = 1.0
        N = 1e5
        if sampling_method == 'crude':
            N = 1e6
        X, y = monte_carlo_simulation(_circle, random_variables, N,
                                      sampling_method=sampling_method)

        estimated = float(y[y<=0].size) / float(y.size) * 4.
        self.assertAlmostEqual(true, estimated, places=2,
                               msg="Monte Carlo integration failed.")

    def test_area_circle_crude(self):
        self.area_circle('crude')

    def test_area_circle_sobol(self):
        self.area_circle('sobol')

    def test_area_circle_latin_random(self):
        self.area_circle('latin_random')

    def test_area_circle_latin_center(self):
        self.area_circle('latin_center')

    def test_area_circle_latin_edge(self):
        self.area_circle('latin_edge')


if __name__ == '__main__':
    unittest.main()