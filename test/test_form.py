# -*- coding: utf-8 -*-
if __package__ is None:
    import sys
    sys.path.append('../')
import unittest
from relsis import *
import numpy as np



class TestFormSolver(unittest.TestCase):
    def setUp(self):
        self.random_variables = [NormalRandomVariable(20., 4.),
                                 NormalRandomVariable(5., 3.)]
        self.limit_state_function = lambda x: x[0] - x[1]
        self.beta_true = (20.-5.) / np.sqrt(4**2+3**2) # beta = 3

    def test_form_solver(self):
        result = form_solver(self.limit_state_function, self.random_variables)
        beta = result['beta']
        self.assertAlmostEqual(beta, self.beta_true)


if __name__ == '__main__':
    unittest.main()