"""Testing suite for DirectPythonInterface.

Created on April 10th 2019
@author: Sebastian Brandstaeter
"""
import numpy as np
import pytest
from pathos.multiprocessing import ProcessingPool as Pool

import pqueens.parameters.parameters as parameters_module
from pqueens.interfaces import from_config_create_interface
from pqueens.interfaces.direct_python_interface import DirectPythonInterface
from pqueens.tests.integration_tests.example_simulator_functions import (
    example_simulator_function_by_name,
)
from pqueens.utils.path_utils import relative_path_from_pqueens
from pqueens.utils.pool_utils import create_pool
import logging
_logger = logging.getLogger(__name__)
@pytest.fixture(scope='module')
def parameters():
    """Options dictionary to create variables."""
    uncertain_parameter = {}
    uncertain_parameter['dimension'] = 1
    uncertain_parameter['distribution'] = "uniform"
    uncertain_parameter['lower_bound'] = -3.14
    uncertain_parameter['upper_bound'] = 3.14

    random_variables = {}
    random_variables['x1'] = uncertain_parameter
    random_variables['x2'] = uncertain_parameter
    random_variables['x3'] = uncertain_parameter

    parameters = {'parameters': {}}
    parameters['parameters']['random_variables'] = random_variables

    parameters_module.from_config_create_parameters(parameters)


@pytest.fixture(scope='module')
def samples():
    """Parameters and samples."""
    # set values
    samples = np.ones((2, 3))
    return samples


@pytest.fixture(scope='module')
def expected_result():
    """Expected result of ishigami funciton for [1., 1., 1.]."""
    return np.array([[5.8821320112036846]])


@pytest.fixture(scope='module')
def expected_results(expected_result):
    """Expected results corresponding to list_of_samples."""
    return np.concatenate([expected_result, expected_result])


@pytest.fixture(scope='module')
def config(parameters):
    """Minimal config dict to create a Direct-Python-Interface."""
    config = {}
    config['test_interface'] = {
        'type': 'direct_python_interface',
        'function_name': 'ishigami90',
    }

    return config


@pytest.fixture(scope='module')
def config_by_path(parameters):
    """Minimal config dict to create a Direct-Python-Interface."""
    path_to_file = relative_path_from_pqueens(
        "tests/integration_tests/example_simulator_functions/ishigami90.py", as_str=True
    )
    _logger.info(path_to_file)
    config = {}
    config['test_interface'] = {
        'type': 'direct_python_interface',
        'function_name': "ishigami90",
        'external_python_module_function': path_to_file,
    }

    return config


@pytest.fixture(scope='module')
def config_parallel(config):
    """Configure parallel evaluation.

    Minimal config dict to create a Direct-Python-Interface with
    parallel evaluation of multiple forward calls.
    """
    config['test_interface']['num_workers'] = 2

    return config


@pytest.fixture(scope='module')
def direct_python_interface(parameters):
    """Direct python interface."""
    interface_name = 'test_interface'
    function = example_simulator_function_by_name("ishigami90")
    pool = None
    return DirectPythonInterface(interface_name, function, pool)


@pytest.fixture(scope='module')
def direct_python_interface_parallel(parameters):
    """An instance of Variables class."""
    interface_name = 'test_interface'
    function = example_simulator_function_by_name("ishigami90")
    pool = create_pool(2)
    return DirectPythonInterface(interface_name, function, pool)


def test_map(samples, expected_results, direct_python_interface):
    """Test mapping from input to response/ output."""
    output = direct_python_interface.evaluate(samples)

    assert direct_python_interface.pool is None
    np.testing.assert_allclose(output["mean"], expected_results)


def test_map_parallel(samples, expected_results, direct_python_interface_parallel):
    """Test parallel mapping.

    Test parallel mapping from multiple input vectors to corresponding
    responses.
    """
    output = direct_python_interface_parallel.evaluate(samples)

    assert isinstance(direct_python_interface_parallel.pool, Pool)
    np.testing.assert_allclose(output["mean"], expected_results)


def test_create_from_config(parameters, config):
    """Given a config dict instantiate DirectPythonInterface."""
    direct_python_interface = from_config_create_interface('test_interface', config)
    # ensure correct types
    assert direct_python_interface.pool is None
    assert isinstance(direct_python_interface, DirectPythonInterface)


def test_create_from_config_parallel(parameters, config_parallel):
    """Test creation from config file.

    Given a config dict instantiate DirectPythonInterface with parallel
    evaluation of multiple forward calls.
    """
    direct_python_interface_parallel = from_config_create_interface(
        'test_interface', config_parallel
    )
    # ensure correct types
    assert isinstance(direct_python_interface_parallel.pool, Pool)
    assert isinstance(direct_python_interface_parallel, DirectPythonInterface)


def test_function_keywords(samples, config, config_by_path):
    """Test interface using by path and by name."""
    direct_python_interface_function_name = from_config_create_interface('test_interface', config)
    direct_python_interface_path = from_config_create_interface('test_interface', config_by_path)
    results_function_name = direct_python_interface_function_name.evaluate(samples)
    results_path = direct_python_interface_path.evaluate(samples)

    np.testing.assert_equal(results_function_name, results_path)
