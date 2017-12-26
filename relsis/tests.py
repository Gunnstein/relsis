# -*- coding: utf-8 -*-
import numpy as np
import unittest
from randomvariables import *
from main import *

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

if __name__ == "__main__":
    unittest.main()

