# -*- coding: utf-8 -*-
if __package__ is None:
    import sys
    sys.path.append('../')
import unittest
from relsis import *
import numpy as np

class MonteCarloSimulationTestCase(unittest.TestCase):
    def _circle(self, x):
        return x[0]**2 + x[1]**2 - 1. / np.pi

    def area_circle(self, sampling_method):
        random_variables = [UniformRandomVariable(0., 1.),
                            UniformRandomVariable(0., 1.)]
        true = 1.0
        N = 1e5
        if sampling_method == 'crude':
            N = 1e6
        X, y = monte_carlo_simulation(self._circle, random_variables, N,
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