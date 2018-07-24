'''
Created on November 20th  2017
@author: jbi

'''
import unittest
import numpy as np
from pqueens.interfaces.direct_python_interface import DirectPythonInterface
from pqueens.models.simulation_model import SimulationModel
from pqueens.variables.variables import Variables
from pqueens.iterators.morris_salib_wrapper_iterator import MorrisSALibIterator



class TestSAMorrisIshigami(unittest.TestCase):
    def setUp(self):

        uncertain_parameters = {}
        uncertain_parameter = {}
        random_variables = {}
        uncertain_parameter["type"] = "FLOAT"
        uncertain_parameter["size"] = 1
        uncertain_parameter["max"] = 3.14159265359
        uncertain_parameter["min"] = -3.14159265359
        #luncertain_parameter["distribution"] = "uniform"
        #uncertain_parameter["distribution_parameter"] = [-3.14159265359,3.14159265359]

        random_variables['x1'] = uncertain_parameter
        random_variables['x2'] = uncertain_parameter
        random_variables['x3'] = uncertain_parameter
        uncertain_parameters["random_variables"] = random_variables

        some_settings = {}
        some_settings["experiment_name"] = "test"

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
                                               num_bootstrap_samples=1000,
                                               result_description=None,
                                               global_settings=some_settings)



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
