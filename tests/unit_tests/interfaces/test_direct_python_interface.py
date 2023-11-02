"""Testing suite for DirectPythonInterface."""
import logging

import numpy as np
import pytest
from pathos.multiprocessing import ProcessingPool as Pool

from queens.distributions.uniform import UniformDistribution
from queens.example_simulator_functions import example_simulator_function_by_name
from queens.example_simulator_functions.ishigami90 import ishigami90
from queens.interfaces.direct_python_interface import DirectPythonInterface
from queens.parameters.parameters import Parameters
from queens.utils.path_utils import relative_path_from_source

_logger = logging.getLogger(__name__)


@pytest.fixture(name="parameters", scope='module')
def fixture_parameters():
    """Options dictionary to create variables."""
    parameter_x1 = UniformDistribution(lower_bound=-3.14, upper_bound=3.14)
    parameter_x2 = UniformDistribution(lower_bound=-3.14, upper_bound=3.14)
    parameter_x3 = UniformDistribution(lower_bound=-3.14, upper_bound=3.14)
    return Parameters(x1=parameter_x1, x2=parameter_x2, x3=parameter_x3)


@pytest.fixture(name="samples", scope='module')
def fixture_samples():
    """Parameters and samples."""
    # set values
    samples = np.ones((2, 3))
    return samples


@pytest.fixture(name="expected_result", scope='module')
def fixture_expected_result():
    """Expected result of ishigami function for [1., 1., 1.]."""
    return np.array([[5.8821320112036846]])


@pytest.fixture(name="expected_results", scope='module')
def fixture_expected_results(expected_result):
    """Expected results corresponding to *list_of_samples*."""
    return np.concatenate([expected_result, expected_result])


@pytest.fixture(name="direct_python_interface", scope='module')
def fixture_direct_python_interface(parameters):
    """Direct python interface."""
    return DirectPythonInterface(parameters=parameters, function="ishigami90", num_workers=1)


@pytest.fixture(name="direct_python_interface_parallel", scope='module')
def fixture_direct_python_interface_parallel(parameters):
    """Direct python interface."""
    return DirectPythonInterface(parameters=parameters, function="ishigami90", num_workers=2)


@pytest.fixture(name="direct_python_interface_function_passing", scope='module')
def fixture_direct_python_interface_function_passing(parameters):
    """Direct python interface with function in init."""
    function = example_simulator_function_by_name("ishigami90")
    return DirectPythonInterface(parameters=parameters, function=function, num_workers=1)


@pytest.fixture(name="function_without_job_id")
def fixture_function_without_job_id():
    """Function without job_id argument."""

    def function(x1, x2, x3):  # pylint: disable=invalid-name
        return ishigami90(x1, x2, x3)

    return function


@pytest.fixture(name="function_with_job_id")
def fixture_function_with_job_id():
    """Function with job_id argument."""

    def function(x1, x2, x3, job_id):  # pylint: disable=invalid-name,unused-argument
        return ishigami90(x1, x2, x3)

    return function


@pytest.fixture(name="function_with_kwargs")
def fixture_function_with_kwargs():
    """Function with kwargs."""

    def function(x1, x2, x3, **kwargs):  # pylint: disable=invalid-name,unused-argument
        if "job_id" not in kwargs:
            raise AttributeError("job_id was not passed in the kwargs")
        return ishigami90(x1, x2, x3)

    return function


@pytest.fixture(name="direct_python_interface_path", scope='module')
def fixture_direct_python_interface_path(parameters):
    """Minimal config dict to create a Direct-Python-Interface."""
    path_to_file = relative_path_from_source("example_simulator_functions/ishigami90.py")
    _logger.info(path_to_file)
    return DirectPythonInterface(
        parameters=parameters, function='ishigami90', external_python_module_function=path_to_file
    )


@pytest.mark.parametrize(
    "function_fixture,expected_function_requires_job_id",
    [
        ("function_with_job_id", True),
        ("function_without_job_id", False),
        ("function_with_kwargs", True),
    ],
)
def test_function_requires_job_id(
    parameters, function_fixture, expected_function_requires_job_id, request
):
    """Test function_requires_job_id is set properly."""
    function = request.getfixturevalue(function_fixture)
    interface = DirectPythonInterface(parameters=parameters, function=function)
    assert interface.function_requires_job_id == expected_function_requires_job_id


@pytest.mark.parametrize(
    "function_fixture",
    [
        "function_with_job_id",
        "function_without_job_id",
        "function_with_kwargs",
    ],
)
def test_evaluate_depending_on_function_requires_job_id(
    parameters, function_fixture, expected_results, request, samples
):
    """Test if function is called properly."""
    function = request.getfixturevalue(function_fixture)
    interface = DirectPythonInterface(parameters=parameters, function=function)
    np.testing.assert_equal(interface.evaluate(samples)["mean"], expected_results)


@pytest.mark.parametrize("expected_called_with_job_id", [True, False])
def test_create_samples_list(samples, expected_called_with_job_id, direct_python_interface):
    """Test create_samples_list."""
    samples_list = direct_python_interface.create_samples_list(
        samples, add_job_id=expected_called_with_job_id
    )
    for sample in samples_list:
        assert ("job_id" in sample) == expected_called_with_job_id


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


def test_init(direct_python_interface):
    """Test init of DirectPythonInterface."""
    # ensure correct types
    assert direct_python_interface.pool is None
    assert isinstance(direct_python_interface, DirectPythonInterface)


def test_create_from_config_parallel(direct_python_interface_parallel):
    """Test DirectPythonInterface with parallel evaluation."""
    # ensure correct types
    assert isinstance(direct_python_interface_parallel.pool, Pool)
    assert isinstance(direct_python_interface_parallel, DirectPythonInterface)


def test_function_keywords(samples, direct_python_interface_path, direct_python_interface):
    """Test interface by path and by name."""
    results_function_name = direct_python_interface_path.evaluate(samples)
    results_path = direct_python_interface.evaluate(samples)

    np.testing.assert_equal(results_function_name, results_path)


def test_function_directly(
    samples, direct_python_interface_function_passing, direct_python_interface
):
    """Test interface by passing a function directly."""
    results_function_name = direct_python_interface_function_passing.evaluate(samples)
    results_path = direct_python_interface.evaluate(samples)

    np.testing.assert_equal(results_function_name, results_path)
