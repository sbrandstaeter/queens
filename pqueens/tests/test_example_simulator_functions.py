'''
Created on June 20th 2017
@author: jbi

'''
import unittest
import pqueens.example_simulator_functions.agawal as agawal

class TestAgawal(unittest.TestCase):

    def setUp(self):
        self.params1 = {'x1': 0.6, 'x2': 0.4}
        self.params2 = {'x1': 0.4, 'x2': 0.4}
        self.dummy_id = 100

    def test_vals_params(self):
        actual_result1 = agawal.main(self.dummy_id, self.params1)
        desired_result1 = 0.0
        self.assertAlmostEqual(actual_result1, desired_result1, places=8,
                               msg=None, delta=None)

        actual_result2 = agawal.main(self.dummy_id, self.params2)
        desired_result2 = 0.90450849718747361
        self.assertAlmostEqual(actual_result2, desired_result2, places=8,
                               msg=None, delta=None)
