'''
Created on Dezember 11th  2017
@author: jbi

'''
import unittest
import numpy as np
from pqueens.interfaces.direct_python_interface import DirectPythonInterface
from pqueens.interfaces.approximation_interface import ApproximationInterface
from pqueens.interfaces.job_interface import JobInterface
from pqueens.interfaces.interface import Interface

#from pqueens.variables.variables import Variables

class TestDirectPythonInterface(unittest.TestCase):
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

class TestApproximationInterface(unittest.TestCase):
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
        self.config['test_interface'] = {'type':'approximation_interface',
                                         'approximation': 'dummy_approximation',
                                         'main_file':'pqueens/example_simulator_functions/ishigami.py'}

        self.config['parameters'] = uncertain_parameters
        self.config['dummy_approximation'] = 'some_stuff'

    def test_construction(self):
        interface = Interface.from_config_create_interface('test_interface', self.config)
        # ensure correct type
        self.assertIsInstance(interface, ApproximationInterface)


class TestJobInterface(unittest.TestCase):
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

        self.config = {}
        self.config['experiment-name'] = 'test-experiment'
        self.config['test_interface'] = {'type':'job_interface',
                                         'resources': 'dummy_resource'}

        self.config['parameters'] = uncertain_parameters

        dummy_resource = {}
        dummy_resource['my_machine'] = {'scheduler': 'local', 'max-concurrent':5,
                                        'max-finished-jobs' : 100}
        self.config['database'] = {}
        self.config['database']['address'] = 'localhost:27017'
        self.config['output_dir'] = {}
        self.config['driver'] = {}
        self.config['driver']['driver_type'] = 'local'
        self.config['driver']['driver_params'] = {}

        self.config['resources'] = {}
        self.config['resources']['dummy_resource'] = dummy_resource

    def test_construction(self):
        interface = Interface.from_config_create_interface('test_interface', self.config)
        # ensure correct type
        self.assertIsInstance(interface, JobInterface)
