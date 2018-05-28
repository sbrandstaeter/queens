'''
Created on November 20th  2017
@author: jbi

'''
import unittest
import numpy as np
from pqueens.interfaces.direct_python_interface import DirectPythonInterface
from pqueens.models.simulation_model import SimulationModel
from pqueens.variables.variables import Variables
from pqueens.iterators.saltelli_iterator import SaltelliIterator
from pqueens.iterators.morris_salib_wrapper_iterator import MorrisSALibIterator

class TestSASaltelliIshigami(unittest.TestCase):
    def setUp(self):
        uncertain_parameters = {}
        uncertain_parameter = {}
        uncertain_parameter["type"] = "FLOAT"
        uncertain_parameter["size"] = 1
        uncertain_parameter["distribution"] = "uniform"
        uncertain_parameter["distribution_parameter"] = [-3.14159265359, 3.14159265359]

        uncertain_parameters['x1'] = uncertain_parameter
        uncertain_parameters['x2'] = uncertain_parameter
        uncertain_parameters['x3'] = uncertain_parameter

        self.variables = Variables.from_uncertain_parameters_create(uncertain_parameters)
        # create interface
        self.interface =  DirectPythonInterface('test_interface',
                                                'pqueens/example_simulator_functions/ishigami.py',
                                                self.variables)

        # create mock model
        self.model = SimulationModel("my_model", self.interface, uncertain_parameters)

        # setup input paramater for init of Saltelli iterator
        # Note, initialization from config dict is done separately
        self.my_iterator = SaltelliIterator(self.model, seed=42, num_samples=3, calc_second_order=True,
                            num_bootstrap_samples=1000, confidence_level=0.95)



    def test_correct_sampling(self):
        """ Test if scaling works correctly """
        self.my_iterator.pre_run()

        # asser that samples match
        ref_vals = np.array([[-1.7610099445, -2.5341362616,  0.1165825399],
                             [ 1.1106020904, -2.5341362616,  0.1165825399],
                             [-1.7610099445, -1.3805827091,  0.1165825399],
                             [-1.7610099445, -2.5341362616,  2.5586799542],
                             [-1.7610099445, -1.3805827091,  2.5586799542],
                             [ 1.1106020904, -2.5341362616,  2.5586799542],
                             [ 1.1106020904, -1.3805827091,  0.1165825399],
                             [ 1.1106020904, -1.3805827091,  2.5586799542],
                             [ 1.3805827091,  0.607456392,  -3.0250101137],
                             [-2.0309905632,  0.607456392,  -3.0250101137],
                             [ 1.3805827091,  1.7610099445, -3.0250101137],
                             [ 1.3805827091,  0.607456392,  -0.5829126994],
                             [ 1.3805827091,  1.7610099445, -0.5829126994],
                             [-2.0309905632,  0.607456392,  -0.5829126994],
                             [-2.0309905632,  1.7610099445, -3.0250101137],
                             [-2.0309905632,  1.7610099445, -0.5829126994],
                             [ 2.9513790359, -0.9633399348,  1.6873788667],
                             [ 2.6813984172, -0.9633399348,  1.6873788667],
                             [ 2.9513790359,  0.1902136177,  1.6873788667],
                             [ 2.9513790359, -0.9633399348, -2.1537090262],
                             [ 2.9513790359,  0.1902136177, -2.1537090262],
                             [ 2.6813984172, -0.9633399348, -2.1537090262],
                             [ 2.6813984172,  0.1902136177,  1.6873788667],
                             [ 2.6813984172,  0.1902136177, -2.1537090262]])

        np.set_printoptions(precision=10)
        #print("self.samples {}".format(self.my_iterator.samples))
        #print("shape scale_samples: {}".format(scaled_samples))
        np.testing.assert_allclose(self.my_iterator.samples,ref_vals, 1e-07, 1e-07)


    def test_correct_sensitivity_indices(self):
        """ Test if we get error when the number os distributions doas not match
            the number of parameters """
        self.my_iterator.pre_run()
        self.my_iterator.core_run()
        si = self.my_iterator.sensitivity_incides
        #self.my_iterator._SaltelliIterator__print_results()
        #print("si {}".format(si))

        # ref vals S1
        ref_s1 = np.array([-0.3431210952, 1.0121533295, -0.8953575936])
        ref_s1_conf = np.array([1.7706135992, 1.0430560342, 0.6186849792])

        # ref vals ST
        ref_st = np.array([3.0745206587, 0.5874782724, 0.8353123287])
        ref_st_conf = np.array([7.9399807422, 1.0490361716, 1.6090408771])

        # ref vals S2
        ref_s2 = np.array([[np.nan, 0.5204364349, 1.6628473529],
                           [np.nan, np.nan, -0.216330859],
                           [np.nan, np.nan, np.nan]])
        ref_s2_conf = np.array([[np.nan, 3.2073680292, 1.2690192793],
                                [np.nan, np.nan, 0.6632167394],
                                [np.nan, np.nan, np.nan]])

        np.testing.assert_allclose(si['S1'], ref_s1, 1e-07, 1e-07)
        np.testing.assert_allclose(si['S1_conf'], ref_s1_conf, 1e-07, 1e-07)

        np.testing.assert_allclose(si['ST'], ref_st, 1e-07, 1e-07)
        np.testing.assert_allclose(si['ST_conf'], ref_st_conf, 1e-07, 1e-07)

        np.testing.assert_allclose(si['S2'], ref_s2, 1e-07, 1e-07)
        np.testing.assert_allclose(si['S2_conf'], ref_s2_conf, 1e-07, 1e-07)

class TestSASaltelliBorehole(unittest.TestCase):
    def setUp(self):
        uncertain_parameters = {}

        #| rw  ~ N(0.10,0.0161812)
        uncertain_parameter1 = {}
        uncertain_parameter1["type"] = "FLOAT"
        uncertain_parameter1["size"] = 1
        uncertain_parameter1["distribution"] = "normal"
        uncertain_parameter1["distribution_parameter"] = [0.1, 0.0161812]

        #| r   ~ Lognormal(7.71,1.0056)
        uncertain_parameter2 = {}
        uncertain_parameter2["type"] = "FLOAT"
        uncertain_parameter2["size"] = 1
        uncertain_parameter2["distribution"] = "lognormal"
        uncertain_parameter2["distribution_parameter"] = [7.71, 1.0056]

        #| Tu  ~ Uniform[63070, 115600]
        uncertain_parameter3 = {}
        uncertain_parameter3["type"] = "FLOAT"
        uncertain_parameter3["size"] = 1
        uncertain_parameter3["distribution"] = "uniform"
        uncertain_parameter3["distribution_parameter"] = [63070, 115600]


        #| Hu  ~ Uniform[990, 1110]
        uncertain_parameter4 = {}
        uncertain_parameter4["type"] = "FLOAT"
        uncertain_parameter4["size"] = 1
        uncertain_parameter4["distribution"] = "uniform"
        uncertain_parameter4["distribution_parameter"] = [990, 1110]

        #| Tl  ~ Uniform[63.1, 116]
        uncertain_parameter5 = {}
        uncertain_parameter5["type"] = "FLOAT"
        uncertain_parameter5["size"] = 1
        uncertain_parameter5["distribution"] = "uniform"
        uncertain_parameter5["distribution_parameter"] = [63.1, 116]

        #| Hl  ~ Uniform[700, 820]
        uncertain_parameter6 = {}
        uncertain_parameter6["type"] = "FLOAT"
        uncertain_parameter6["size"] = 1
        uncertain_parameter6["distribution"] = "uniform"
        uncertain_parameter6["distribution_parameter"] = [700, 820]

        #| L   ~ Uniform[1120, 1680]
        uncertain_parameter7 = {}
        uncertain_parameter7["type"] = "FLOAT"
        uncertain_parameter7["size"] = 1
        uncertain_parameter7["distribution"] = "uniform"
        uncertain_parameter7["distribution_parameter"] = [1120, 1680]


        #| Kw  ~ Uniform[9855, 12045]
        uncertain_parameter8 = {}
        uncertain_parameter8["type"] = "FLOAT"
        uncertain_parameter8["size"] = 1
        uncertain_parameter8["distribution"] = "uniform"
        uncertain_parameter8["distribution_parameter"] = [9855, 12045]

        uncertain_parameters['rw'] = uncertain_parameter1
        uncertain_parameters['r'] = uncertain_parameter2
        uncertain_parameters['Tu'] = uncertain_parameter3
        uncertain_parameters['Hu'] = uncertain_parameter4
        uncertain_parameters['Tl'] = uncertain_parameter5
        uncertain_parameters['Hl'] = uncertain_parameter6
        uncertain_parameters['L'] = uncertain_parameter7
        uncertain_parameters['Kw'] = uncertain_parameter8

        self.variables = Variables.from_uncertain_parameters_create(uncertain_parameters)
        # create interface
        self.interface =  DirectPythonInterface('test_interface',
                                                'pqueens/example_simulator_functions/borehole_hifi.py',
                                                self.variables)

        # create mock model
        self.model = SimulationModel("my_model", self.interface, uncertain_parameters)

        # setup input paramater for init of Saltelli iterator
        # Note, initialization from config dict is done separately
        self.my_iterator = SaltelliIterator(self.model, seed=42, num_samples=3, calc_second_order=False,
                            num_bootstrap_samples=1000, confidence_level=0.95)

    def test_correct_sampling(self):
        """ Test if scaling works correctly """
        self.my_iterator.pre_run()

        # asser that samples matc
        ref_vals = np.array([[8.7490038761e-02, 6.0305646426e+02, 9.0309677734e+04,
                1.0712109375e+03,   7.7926464844e+01,   8.0886718750e+02,
                1.1457031250e+03,   1.1824716797e+04],
             [  1.0003960966e-01,   6.0305646426e+02,   9.0309677734e+04,
                1.0712109375e+03,   7.7926464844e+01,   8.0886718750e+02,
                1.1457031250e+03,   1.1824716797e+04],
             [  8.7490038761e-02,   5.0318242770e+02,   9.0309677734e+04,
                1.0712109375e+03,   7.7926464844e+01,   8.0886718750e+02,
                1.1457031250e+03,   1.1824716797e+04],
             [  8.7490038761e-02,   6.0305646426e+02,   6.7532998047e+04,
                1.0712109375e+03,   7.7926464844e+01,   8.0886718750e+02,
                1.1457031250e+03,   1.1824716797e+04],
             [  8.7490038761e-02,   6.0305646426e+02,   9.0309677734e+04,
                1.0205859375e+03,   7.7926464844e+01,   8.0886718750e+02,
                1.1457031250e+03,   1.1824716797e+04],
             [  8.7490038761e-02,   6.0305646426e+02,   9.0309677734e+04,
                1.0712109375e+03,   7.1623925781e+01,   8.0886718750e+02,
                1.1457031250e+03,   1.1824716797e+04],
             [  8.7490038761e-02,   6.0305646426e+02,   9.0309677734e+04,
                1.0712109375e+03,   7.7926464844e+01,   7.4605468750e+02,
                1.1457031250e+03,   1.1824716797e+04],
             [  8.7490038761e-02,   6.0305646426e+02,   9.0309677734e+04,
                1.0712109375e+03,   7.7926464844e+01,   8.0886718750e+02,
                1.2003906250e+03,   1.1824716797e+04],
             [  8.7490038761e-02,   6.0305646426e+02,   9.0309677734e+04,
                1.0712109375e+03,   7.7926464844e+01,   8.0886718750e+02,
                1.1457031250e+03,   1.0665556641e+04],
             [  1.0003960966e-01,   5.0318242770e+02,   6.7532998047e+04,
                1.0205859375e+03,   7.1623925781e+01,   7.4605468750e+02,
                1.2003906250e+03,   1.0665556641e+04],
             [  1.0941793420e-01,   2.8530183666e+03,   6.4044677734e+04,
                1.0112109375e+03,   1.0437646484e+02,   7.4886718750e+02,
                1.4257031250e+03,   1.0729716797e+04],
             [  4.9882469592e-02,   2.8530183666e+03,   6.4044677734e+04,
                1.0112109375e+03,   1.0437646484e+02,   7.4886718750e+02,
                1.4257031250e+03,   1.0729716797e+04],
             [  1.0941793420e-01,   2.6588846104e+03,   6.4044677734e+04,
                1.0112109375e+03,   1.0437646484e+02,   7.4886718750e+02,
                1.4257031250e+03,   1.0729716797e+04],
             [  1.0941793420e-01,   2.8530183666e+03,   9.3797998047e+04,
                1.0112109375e+03,   1.0437646484e+02,   7.4886718750e+02,
                1.4257031250e+03,   1.0729716797e+04],
             [  1.0941793420e-01,   2.8530183666e+03,   6.4044677734e+04,
                1.0805859375e+03,   1.0437646484e+02,   7.4886718750e+02,
                1.4257031250e+03,   1.0729716797e+04],
             [  1.0941793420e-01,   2.8530183666e+03,   6.4044677734e+04,
                1.0112109375e+03,   9.8073925781e+01,   7.4886718750e+02,
                1.4257031250e+03,   1.0729716797e+04],
             [  1.0941793420e-01,   2.8530183666e+03,   6.4044677734e+04,
                1.0112109375e+03,   1.0437646484e+02,   8.0605468750e+02,
                1.4257031250e+03,   1.0729716797e+04],
             [  1.0941793420e-01,   2.8530183666e+03,   6.4044677734e+04,
                1.0112109375e+03,  1.0437646484e+02,   7.4886718750e+02,
                1.4803906250e+03,   1.0729716797e+04],
             [  1.0941793420e-01,  2.8530183666e+03,   6.4044677734e+04,
                1.0112109375e+03,   1.0437646484e+02,   7.4886718750e+02,
                1.4257031250e+03,   1.1760556641e+04],
             [  4.9882469592e-02,   2.6588846104e+03,   9.3797998047e+04,
                1.0805859375e+03,   9.8073925781e+01,   8.0605468750e+02,
                1.4803906250e+03,   1.1760556641e+04],
             [  1.3036871512e-01,   1.5004069698e+03,   1.0344217773e+05,
                1.1012109375e+03,   9.1151464844e+01,  7.1886718750e+02,
                1.2857031250e+03,   1.0182216797e+04],
             [  8.9135621682e-02,   1.5004069698e+03,   1.0344217773e+05,
                1.1012109375e+03,   9.1151464844e+01,   7.1886718750e+02,
                1.2857031250e+03,   1.0182216797e+04],
             [  1.3036871512e-01,   5.5856113864e+03,   1.0344217773e+05,
                1.1012109375e+03,   9.1151464844e+01,   7.1886718750e+02,
                1.2857031250e+03,   1.0182216797e+04],
             [  1.3036871512e-01,   1.5004069698e+03,  1.0693049805e+05,
                1.1012109375e+03,   9.1151464844e+01,   7.1886718750e+02,
                1.2857031250e+03,   1.0182216797e+04],
             [  1.3036871512e-01,   1.5004069698e+03,   1.0344217773e+05,
                1.0505859375e+03,   9.1151464844e+01,   7.1886718750e+02,
                1.2857031250e+03,   1.0182216797e+04],
             [  1.3036871512e-01,   1.5004069698e+03,   1.0344217773e+05,
                1.1012109375e+03,   8.4848925781e+01,   7.1886718750e+02,
                1.2857031250e+03,   1.0182216797e+04],
             [  1.3036871512e-01,   1.5004069698e+03,   1.0344217773e+05,
                1.1012109375e+03,   9.1151464844e+01,   7.1605468750e+02,
                1.2857031250e+03,   1.0182216797e+04],
             [  1.3036871512e-01,   1.5004069698e+03,   1.0344217773e+05,
                1.1012109375e+03,   9.1151464844e+01,   7.1886718750e+02,
                1.6203906250e+03,   1.0182216797e+04],
             [  1.3036871512e-01,   1.5004069698e+03,   1.0344217773e+05,
                1.1012109375e+03,   9.1151464844e+01,   7.1886718750e+02,
                1.2857031250e+03,   1.0118056641e+04],
             [  8.9135621682e-02,   5.5856113864e+03,   1.0693049805e+05,
                1.0505859375e+03,   8.4848925781e+01,   7.1605468750e+02,
                1.6203906250e+03,   1.0118056641e+04]])


        np.set_printoptions(precision=10)
        #print("self.samples {}".format(self.my_iterator.samples))
        #print("shape scale_samples: {}".format(scaled_samples))
        np.testing.assert_allclose(self.my_iterator.samples,ref_vals, 1e-07, 1e-07)

    def test_correct_sensitivity_indices(self):
        """ Test if we get error when the number os distributions doas not match
            the number of parameters """
        self.my_iterator.pre_run()
        self.my_iterator.core_run()
        si = self.my_iterator.sensitivity_incides
        #self.my_iterator._SaltelliIteratorNew__print_results()
        #print("si {}".format(si))

        # ref vals S1
        ref_s1 = np.array([1.2523251854e+00, 9.6453674593e-04, -2.1378397687e-06,
                           -8.2098449994e-02, 8.5052704785e-04, 1.6245550264e-01,
                           2.5538377341e-01, -6.9145165798e-02])

        #ref_s1_conf
        ref_s1_conf = np.array([4.2476267297e+00, 1.7763349958e-03, 1.5755264507e-05,
                                2.2067844861e+00, 3.2707543654e-03, 2.4596705025e+00,
                                3.3638575489e-01, 1.0162718426e+00])
        # ref vals ST
        ref_st = np.array([9.7956424550e-01, 2.1201464446e-06, 3.1797806991e-12,
                           8.7564004034e-02, 6.8573143286e-07, 4.4369543936e-02,
                           9.7574648954e-02, 8.0458830639e-03])

        ref_st_conf = np.array([1.8947540914e+00, 2.5825608751e-06, 3.9874144570e-11,
                                8.5489821668e-01, 3.4512403037e-06, 1.3511745947e+00,
                                1.1211206810e-01, 2.2449074345e-01])
        # ref vals S2
        np.testing.assert_allclose(si['S1'], ref_s1, 1e-07, 1e-07)
        np.testing.assert_allclose(si['S1_conf'], ref_s1_conf, 1e-07, 1e-07)

        np.testing.assert_allclose(si['ST'], ref_st, 1e-07, 1e-07)
        np.testing.assert_allclose(si['ST_conf'], ref_st_conf, 1e-07, 1e-07)

class TestSAMorrisIshigami(unittest.TestCase):
    def setUp(self):
        uncertain_parameters = {}
        uncertain_parameter = {}
        uncertain_parameter["type"] = "FLOAT"
        uncertain_parameter["size"] = 1
        uncertain_parameter["min"] = -3.14159265359
        uncertain_parameter["max"] = 3.14159265359

        uncertain_parameters['x1'] = uncertain_parameter
        uncertain_parameters['x2'] = uncertain_parameter
        uncertain_parameters['x3'] = uncertain_parameter

        self.variables = Variables.from_uncertain_parameters_create(uncertain_parameters)
        # create interface
        self.interface =  DirectPythonInterface('test_interface',
                                                'pqueens/example_simulator_functions/ishigami.py',
                                                self.variables)

        # create mock model
        self.model = SimulationModel("my_model", self.interface, uncertain_parameters)

        # setup input paramater for init of Morris iterator
        self.my_iterator = MorrisSALibIterator(self.model,
                                               num_trajectories=20,
                                               local_optimization=False,
                                               num_optimal_trajectories=4,
                                               grid_jump=2,
                                               num_levels=4,
                                               seed=42,
                                               confidence_level=0.95,
                                               num_bootstrap_samples=1000)



    def test_correct_sampling(self):
        """ Test if sampling works correctly """
        self.my_iterator.pre_run()

        # asser that samples match
        ref_vals = np.array([[1.0471975512, 1.0471975512, -3.1415926536],
                             [-3.1415926536, 1.0471975512, -3.1415926536],
                             [-3.1415926536, -3.1415926536, -3.1415926536],
                             [-3.1415926536, -3.1415926536, 1.0471975512],
                             [-3.1415926536, -1.0471975512, 1.0471975512],
                             [-3.1415926536, 3.1415926536, 1.0471975512],
                             [-3.1415926536, 3.1415926536, -3.1415926536],
                             [1.0471975512, 3.1415926536, -3.1415926536],
                             [3.1415926536, -1.0471975512, 3.1415926536],
                             [3.1415926536, 3.1415926536, 3.1415926536],
                             [-1.0471975512, 3.1415926536, 3.1415926536],
                             [-1.0471975512, 3.1415926536, -1.0471975512],
                             [-1.0471975512, -3.1415926536, 3.1415926536],
                             [3.1415926536, -3.1415926536, 3.1415926536],
                             [3.1415926536, 1.0471975512, 3.1415926536],
                             [3.1415926536, 1.0471975512, -1.0471975512]])
    
        np.set_printoptions(precision=10)
        #print("self.samples {}".format(self.my_iterator.samples))
        #print("shape scale_samples: {}".format(scaled_samples))
        np.testing.assert_allclose(self.my_iterator.samples,ref_vals, 1e-07, 1e-07)


    def test_correct_sensitivity_indices(self):
        """ Test if we get error when the number os distributions doas not match
            the number of parameters """
        self.my_iterator.pre_run()
        self.my_iterator.core_run()
        si = self.my_iterator.si
        #self.my_iterator._MorrisSALibIterator__print_results()

        #print("si  {}".format(si))
        #print(" ")
        #print("si mu_star {}".format(si['mu_star']))
        #print(" ")
        #print("si mu_star_conf {}".format(si['mu_star_conf']))
        #print(" ")
        #print("si sigma {}".format(si['sigma']))

        # ref vals
        ref_mu = np.array([13.9528502149, 0., -3.1243980517])

        ref_mu_star = np.array([13.9528502149, 7.875, 3.1243980517])

        ref_mu_star_conf = np.array([6.9666750666351737e-15,
                                     3.4833375333175868e-15,
                                     5.379288e+00])

        ref_sigma = np.array([0.,9.0932667397,6.248796])

        np.testing.assert_allclose(si['mu'], ref_mu, 1e-07, 1e-07)
        np.testing.assert_allclose(si['mu_star'], ref_mu_star, 1e-07, 1e-07)
        np.testing.assert_allclose(si['mu_star_conf'], ref_mu_star_conf, 1e-07, 1e-07)
        np.testing.assert_allclose(si['sigma'], ref_sigma, 1e-07, 1e-07)
