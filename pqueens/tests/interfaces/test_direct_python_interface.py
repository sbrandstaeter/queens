'''
Created on November 20th  2017
@author: jbi

'''
import unittest
import numpy as np
from pqueens.interfaces.direct_python_interface import DirectPythonInterface
from pqueens.variables.variables import Variables
from pqueens.interfaces.interface import Interface


class TestDirectPythonInterface(unittest.TestCase):
    def setUp(self):

        uncertain_parameters = {}
        uncertain_parameter = {}
        uncertain_parameter["type"] = "FLOAT"
        uncertain_parameter["size"] = 1
        uncertain_parameter["distribution"] = "uniform"
        uncertain_parameter["distribution_parameter"] = [-3.14159265359,3.14159265359]

        uncertain_parameters['x1'] = uncertain_parameter
        uncertain_parameters['x2'] = uncertain_parameter
        uncertain_parameters['x3'] = uncertain_parameter

        self.variables = Variables.from_uncertain_parameters_create(uncertain_parameters)
        # create interface
        self.interface =  DirectPythonInterface('test_interface','pqueens/example_simulator_functions/ishigami.py',self.variables)

    def test_mapping(self):
        """ Test if mapping works correctly """
        # create samples
        self.variables.variables['x1']['value'] = 1.0
        self.variables.variables['x2']['value'] = 1.0
        self.variables.variables['x3']['value'] = 1.0

        my_samples = self.variables
        ref_vals = np.array([[5.8821320112036846]])

        output = self.interface.map([my_samples])
        np.testing.assert_allclose(output["mean"],ref_vals, 1e-09, 1e-09)

class TestDirectPythonInterfaceInitialization(unittest.TestCase):
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


        #self.variables = Variables.from_uncertain_parameters_create(uncertain_parameters)
        self.config = {}
        self.config['test_interface'] = {'type':'direct_python_interface',
                                         'main_file':'pqueens/example_simulator_functions/ishigami.py'}

        self.config['parameters'] = uncertain_parameters

    def test_construction(self):
        interface = Interface.from_config_create_interface('test_interface', self.config)
        # ensure correct type
        self.assertIsInstance(interface, DirectPythonInterface)
