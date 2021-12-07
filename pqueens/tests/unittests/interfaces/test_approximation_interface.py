"""Created on Dezember 11th  2017.

@author: jbi
"""
import unittest

import mock
import pytest

from pqueens.interfaces.approximation_interface import ApproximationInterface
from pqueens.interfaces.interface import Interface


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

        # self.variables = Variables.from_uncertain_parameters_create(uncertain_parameters)
        self.config = {}
        self.config['test_interface'] = {
            'type': 'approximation_interface',
            'approximation': 'dummy_approximation',
            'main_file': 'pqueens/example_simulator_functions/ishigami.py',
        }

        self.config['parameters'] = uncertain_parameters
        self.config['dummy_approximation'] = 'some_stuff'

    @pytest.mark.unit_tests
    def test_construction(self):
        interface = Interface.from_config_create_interface('test_interface', self.config)
        # ensure correct type
        self.assertIsInstance(interface, ApproximationInterface)
