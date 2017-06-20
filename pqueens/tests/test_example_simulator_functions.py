'''
Created on June 20th 2017
@author: jbi

'''
import unittest
import pqueens.example_simulator_functions.agawal as agawal
import pqueens.example_simulator_functions.branin_hifi  as branin_hifi
import pqueens.example_simulator_functions.branin_medfi as branin_medfi
import pqueens.example_simulator_functions.branin_lofi  as branin_lofi

import pqueens.example_simulator_functions.perdikaris_1dsin_lofi  as perdikaris_1dsin_lofi
import pqueens.example_simulator_functions.perdikaris_1dsin_hifi  as perdikaris_1dsin_hifi

import pqueens.example_simulator_functions.park91a_hifi  as park91a_hifi
import pqueens.example_simulator_functions.park91a_lofi  as park91a_lofi

import pqueens.example_simulator_functions.park91b_hifi  as park91b_hifi
import pqueens.example_simulator_functions.park91b_lofi  as park91b_lofi

import pqueens.example_simulator_functions.oakley_ohagan2004  as oakley_ohagan2004

import pqueens.example_simulator_functions.ma2009  as ma2009

import pqueens.example_simulator_functions.currin88_hifi  as currin88_hifi
import pqueens.example_simulator_functions.currin88_lofi  as currin88_lofi



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


class TestPerdikarisMultiFidelity(unittest.TestCase):

    def setUp(self):
        self.params1 = {'x': 0.6}
        self.dummy_id = 100

    def test_vals_params(self):
        actual_result_hifi = perdikaris_1dsin_lofi.main(self.dummy_id, self.params1)
        actual_result_lofi = perdikaris_1dsin_hifi.main(self.dummy_id, self.params1)

        #print("actual_result_hifi {}".format(actual_result_hifi))
        #print("actual_result_lofi {}".format(actual_result_lofi))

        desired_result_hifi  = 0.5877852522924737
        desired_result_lofi  = -0.2813038672746218

        self.assertAlmostEqual(actual_result_hifi, desired_result_hifi,
                               places=8, msg=None, delta=None)

        self.assertAlmostEqual(actual_result_lofi, desired_result_lofi,
                               places=8, msg=None, delta=None)

class TestBraninMultiFidelity(unittest.TestCase):

    def setUp(self):
        self.params1 = {'x1': -4, 'x2': 5}
        self.dummy_id = 100

    def test_vals_params(self):
        actual_result_hifi = branin_hifi.main(self.dummy_id, self.params1)
        actual_result_medfi = branin_medfi.main(self.dummy_id, self.params1)
        actual_result_lofi = branin_lofi.main(self.dummy_id, self.params1)

        #print("actual_result_hifi {}".format(actual_result_hifi))
        #print("actual_result_medfi {}".format(actual_result_medfi))
        #print("actual_result_lofi {}".format(actual_result_lofi))

        desired_result_hifi  = 92.70795679406056
        desired_result_medfi = 125.49860898539086
        desired_result_lofi  = 1.4307273110713652

        self.assertAlmostEqual(actual_result_hifi, desired_result_hifi,
                               places=8, msg=None, delta=None)

        self.assertAlmostEqual(actual_result_medfi, desired_result_medfi,
                               places=8, msg=None, delta=None)

        self.assertAlmostEqual(actual_result_lofi, desired_result_lofi,
                               places=8, msg=None, delta=None)

class TestPark91aMultiFidelity(unittest.TestCase):

    def setUp(self):
        self.params1 = {'x1': 0.3,'x2': 0.6,'x3': 0.5,'x4': 0.1}
        self.dummy_id = 100

    def test_vals_params(self):
        actual_result_hifi = park91a_hifi.main(self.dummy_id, self.params1)
        actual_result_lofi = park91a_lofi.main(self.dummy_id, self.params1)

        #print("actual_result_hifi {}".format(actual_result_hifi))
        #print("actual_result_lofi {}".format(actual_result_lofi))

        desired_result_hifi  = 2.6934187033863846
        desired_result_lofi  = 3.2830146685714103

        self.assertAlmostEqual(actual_result_hifi, desired_result_hifi,
                               places=8, msg=None, delta=None)

        self.assertAlmostEqual(actual_result_lofi, desired_result_lofi,
                               places=8, msg=None, delta=None)

class TestPark91bMultiFidelity(unittest.TestCase):

    def setUp(self):
        self.params1 = {'x1': 0.3,'x2': 0.6,'x3': 0.5,'x4': 0.1}
        self.dummy_id = 100

    def test_vals_params(self):
        actual_result_hifi = park91b_hifi.main(self.dummy_id, self.params1)
        actual_result_lofi = park91b_lofi.main(self.dummy_id, self.params1)

        #print("actual_result_hifi {}".format(actual_result_hifi))
        #print("actual_result_lofi {}".format(actual_result_lofi))

        desired_result_hifi  = 2.091792853577546
        desired_result_lofi  = 1.510151424293055

        self.assertAlmostEqual(actual_result_hifi, desired_result_hifi,
                               places=8, msg=None, delta=None)

        self.assertAlmostEqual(actual_result_lofi, desired_result_lofi,
                               places=8, msg=None, delta=None)

class TestOakleyOHagan(unittest.TestCase):

    def setUp(self):
        self.params1 = {'x1': 0.3,'x2': 0.6,'x3': 0.5,'x4': 0.1,'x5': 0.9,
                        'x6': 0.3,'x7': 0.6,'x8': 0.5,'x9': 0.1,'x10': 0.9,
                        'x11': 0.3,'x12': 0.6,'x13': 0.5,'x14': 0.1,'x15': 0.9}
        self.dummy_id = 100

    def test_vals_params(self):
        actual_result = oakley_ohagan2004.main(self.dummy_id, self.params1)

        #print("actual_result {}".format(actual_result))

        desired_result  = 24.496726490699082

        self.assertAlmostEqual(actual_result, desired_result,
                               places=8, msg=None, delta=None)

class TestMa(unittest.TestCase):

    def setUp(self):
        self.params1 = {'x1': 0.25,'x2': 0.5}
        self.dummy_id = 100

    def test_vals_params(self):
        actual_result = ma2009.main(self.dummy_id, self.params1)

        #print("actual_result {}".format(actual_result))

        desired_result  = 8.8888888888888875

        self.assertAlmostEqual(actual_result, desired_result,
                               places=8, msg=None, delta=None)

class TestCurrin88bMultiFidelity(unittest.TestCase):

    def setUp(self):
        self.params1 = {'x1': 0.6,'x2': 0.1}
        self.dummy_id = 100

    def test_vals_params(self):
        actual_result_hifi = currin88_hifi.main(self.dummy_id, self.params1)
        actual_result_lofi = currin88_lofi.main(self.dummy_id, self.params1)

        #print("actual_result_hifi {}".format(actual_result_hifi))
        #print("actual_result_lofi {}".format(actual_result_lofi))

        desired_result_hifi  = 11.06777716201019
        desired_result_lofi  = 10.964538831722423

        self.assertAlmostEqual(actual_result_hifi, desired_result_hifi,
                               places=8, msg=None, delta=None)

        self.assertAlmostEqual(actual_result_lofi, desired_result_lofi,
                               places=8, msg=None, delta=None)
