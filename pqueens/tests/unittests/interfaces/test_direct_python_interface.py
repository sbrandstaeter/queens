"""Testing suite for DirectPythonInterface.

Created on April 10th 2019
@author: Sebastian Brandstaeter
"""
import multiprocessing

import numpy as np
import pytest

from pqueens.interfaces.direct_python_interface import DirectPythonInterface
from pqueens.interfaces.interface import Interface
from pqueens.variables.variables import Variables


@pytest.fixture(scope='module')
def uncertain_parameters():
    """Options dictionary to create variables."""

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
    """An instance of Variables class."""

    variables = Variables.from_uncertain_parameters_create(uncertain_parameters)

    # set values
    data_vector = np.ones(3)
    variables.update_variables_from_vector(data_vector)
    return variables


@pytest.fixture(scope='module')
def list_of_variables(variables):
    """A list of multiple Variables instances."""

    return [variables, variables]


@pytest.fixture(scope='module')
def expected_result():
    """Expected result of ishigami funciton for [1., 1., 1.]."""

    return np.array([[5.8821320112036846]])


@pytest.fixture(scope='module')
def expected_results(expected_result):
    """Expected results corresponding to list_of_variables."""

    return np.concatenate([expected_result, expected_result])


@pytest.fixture(scope='module')
def config(uncertain_parameters):
    """Minimal config dict to create a Direct-Python-Interface."""

    config = {}
    config['test_interface'] = {'type': 'direct_python_interface', 'main_file': 'ishigami.py'}

    config['parameters'] = uncertain_parameters

    return config


@pytest.fixture(scope='module')
def config_parallel(config):
    """Minimal config dict to create a Direct-Python-Interface with parallel
    evaluation of multiple forward calls."""

    config['test_interface']['num_workers'] = 2

    return config


@pytest.fixture(scope='module')
def direct_python_interface(variables):
    """An instance of Variables class."""

    interface_name = 'test_interface'
    python_function_name = 'ishigami.py'

    return DirectPythonInterface(interface_name, python_function_name, variables, num_workers=1)


@pytest.fixture(scope='module')
def direct_python_interface_parallel(variables):
    """An instance of Variables class."""

    interface_name = 'test_interface'
    python_function_name = 'ishigami.py'

    return DirectPythonInterface(interface_name, python_function_name, variables, num_workers=2)


@pytest.mark.unit_tests
def test_map(list_of_variables, expected_results, direct_python_interface):
    """Test mapping from input to response/ output."""

    output = direct_python_interface.map(list_of_variables)

    assert direct_python_interface.pool is None
    np.testing.assert_allclose(output["mean"], expected_results)


@pytest.mark.unit_tests
def test_map_parallel(list_of_variables, expected_results, direct_python_interface_parallel):
    """Test parallel mapping from multiple input vectors to corresponding
    responses."""

    output = direct_python_interface_parallel.map(list_of_variables)

    assert isinstance(direct_python_interface_parallel.pool, multiprocessing.pool.Pool)
    np.testing.assert_allclose(output["mean"], expected_results)


@pytest.mark.unit_tests
def test_create_from_config(variables, config):
    """Given a config dict instantiate DirectPythonInterface."""

    direct_python_interface = Interface.from_config_create_interface('test_interface', config)
    # ensure correct types
    assert direct_python_interface.pool is None
    assert isinstance(direct_python_interface, DirectPythonInterface)


@pytest.mark.unit_tests
def test_create_from_config_parallel(variables, config_parallel):
    """Given a config dict instantiate DirectPythonInterface with parallel
    evaluation of multiple forward calls."""

    direct_python_interface_parallel = Interface.from_config_create_interface(
        'test_interface', config_parallel
    )
    # ensure correct types
    assert isinstance(direct_python_interface_parallel.pool, multiprocessing.pool.Pool)
    assert isinstance(direct_python_interface_parallel, DirectPythonInterface)
