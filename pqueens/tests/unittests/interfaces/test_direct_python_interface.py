'''
Testing suite for DirectPythonInterface

Created on April 10th 2019
@author: Sebastian Brandstaeter

'''
import numpy as np
import pytest

from pqueens.interfaces.direct_python_interface import DirectPythonInterface
from pqueens.variables.variables import Variables
from pqueens.interfaces.interface import Interface


@pytest.fixture(scope='module')
def uncertain_parameters():
    """ Options dictionary to create variables. """

    uncertain_parameter = {}
    uncertain_parameter['type'] = "FLOAT"
    uncertain_parameter['size'] = 1
    uncertain_parameter['distribution'] = "uniform"
    uncertain_parameter['distribution_parameter'] = [-3.14, 3.14]
    uncertain_parameter['value'] = 1.0

    random_variables = {}
    random_variables['x1'] = uncertain_parameter
    random_variables['x2'] = uncertain_parameter
    random_variables['x3'] = uncertain_parameter

    uncertain_parameters = {}
    uncertain_parameters['random_variables'] = random_variables

    return uncertain_parameters


@pytest.fixture(scope='module')
def variables(uncertain_parameters):
    """ An instance of Variables class. """

    variables = Variables.from_uncertain_parameters_create(uncertain_parameters)

    # set values
    data_vector = np.ones(3)
    variables.update_variables_from_vector(data_vector)
    return  variables


@pytest.fixture(scope='module')
def direct_python_interface(variables):
    """ An instance of Variables class. """

    interface_name = 'test_interface'
    python_function_name = 'ishigami.py'
    return DirectPythonInterface(interface_name, python_function_name, variables)

@pytest.fixture(scope='module')
def config(uncertain_parameters):
    """ Minimal configuration dictionary to create an interface. """
    config = {}
    config['test_interface'] = {'type':'direct_python_interface',
                                'main_file':'ishigami.py'}

    config['parameters'] = uncertain_parameters

    return config


def test_map(variables, direct_python_interface):
    """ Test mapping from input to response/ output. """
    ref_vals = np.array([[5.8821320112036846]])

    output = direct_python_interface.map([variables])
    np.testing.assert_allclose(output["mean"],ref_vals, 1e-09, 1e-09)


def test_create_from_config(variables, config):
    """ Given a config dict instantiate DirectPythonInterface. """

    interface = Interface.from_config_create_interface('test_interface', config)
    # ensure correct type
    assert isinstance(interface, DirectPythonInterface)
