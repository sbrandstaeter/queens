"""TODO_doc."""
from copy import deepcopy

import numpy as np
import pytest
from mock import Mock

from queens.distributions.uniform import UniformDistribution
from queens.iterators.grid_iterator import GridIterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters


@pytest.fixture(name="grid_dict_one")
def fixture_grid_dict_one():
    """TODO_doc."""
    axis_description = {"num_grid_points": 5, "axis_type": "lin", "data_type": "FLOAT"}
    grid_dict_dummy = {"x1": axis_description}
    return grid_dict_dummy


@pytest.fixture(name="grid_dict_two")
def fixture_grid_dict_two():
    """TODO_doc."""
    axis_description = {"num_grid_points": 5, "axis_type": "lin", "data_type": "FLOAT"}
    grid_dict_dummy = {"x1": axis_description, "x2": axis_description}
    return grid_dict_dummy


@pytest.fixture(name="grid_dict_three")
def fixture_grid_dict_three():
    """TODO_doc."""
    axis_description = {"num_grid_points": 5, "axis_type": "lin", "data_type": "FLOAT"}
    grid_dict_dummy = {"x1": axis_description, "x2": axis_description, "x3": axis_description}
    return grid_dict_dummy


@pytest.fixture(name="parameters_one")
def fixture_parameters_one():
    """TODO_doc."""
    rv = UniformDistribution(lower_bound=-2, upper_bound=2)
    return Parameters(x1=rv)


@pytest.fixture(name="parameters_two")
def fixture_parameters_two():
    """TODO_doc."""
    rv = UniformDistribution(lower_bound=-2, upper_bound=2)
    return Parameters(x1=rv, x2=deepcopy(rv))


@pytest.fixture(name="parameters_three")
def fixture_parameters_three():
    """TODO_doc."""
    rv = UniformDistribution(lower_bound=-2, upper_bound=2)
    return Parameters(x1=rv, x2=deepcopy(rv), x3=deepcopy(rv))


@pytest.fixture(name="result_description")
def fixture_result_description():
    """TODO_doc."""
    description = {"write_results": True}
    return description


@pytest.fixture(name="expected_samples_one")
def fixture_expected_samples_one():
    """TODO_doc."""
    x1 = np.linspace(-2, 2, 5)
    return np.atleast_2d(x1).T


@pytest.fixture(name="expected_samples_two")
def fixture_expected_samples_two():
    """TODO_doc."""
    x1 = np.linspace(-2, 2, 5)
    x2 = np.linspace(-2, 2, 5)
    X1, X2 = np.meshgrid(x1, x2)
    samples = np.array([X1.flatten(), X2.flatten()]).T
    return samples


@pytest.fixture(name="expected_samples_three")
def fixture_expected_samples_three():
    """TODO_doc."""
    x1 = np.linspace(-2, 2, 5)
    x2 = np.linspace(-2, 2, 5)
    x3 = np.linspace(-2, 2, 5)
    X1, X2, X3 = np.meshgrid(x1, x2, x3)
    samples = np.array([X1.flatten(), X2.flatten(), X3.flatten()]).T
    return samples


# fixtures for some objects
@pytest.fixture(name="default_model")
def fixture_default_model(parameters_two):
    """TODO_doc."""
    interface = 'dummy_interface'
    model = SimulationModel(interface)
    return model


@pytest.fixture(name="default_grid_iterator")
def fixture_default_grid_iterator(
    _initialize_global_settings, grid_dict_two, parameters_two, default_model, result_description
):
    """TODO_doc."""
    # create iterator object
    my_grid_iterator = GridIterator(
        model=default_model,
        parameters=parameters_two,
        result_description=result_description,
        grid_design=grid_dict_two,
    )
    return my_grid_iterator


# -------------- actual unit_tests --------------------------------------------------
def test_init(
    mocker,
    _initialize_global_settings,
    grid_dict_two,
    parameters_two,
    default_model,
    result_description,
):
    """TODO_doc."""
    # some default input for testing
    num_parameters = 2
    mp = mocker.patch('queens.iterators.iterator.Iterator.__init__')
    GridIterator.parameters = Mock()
    GridIterator.parameters.num_parameters = num_parameters

    my_grid_iterator = GridIterator(
        model=default_model,
        parameters=parameters_two,
        result_description=result_description,
        grid_design=grid_dict_two,
    )

    # tests / asserts
    mp.assert_called_once_with(default_model, parameters_two)
    assert my_grid_iterator.grid_dict == grid_dict_two
    assert my_grid_iterator.result_description == result_description
    assert my_grid_iterator.samples is None
    assert my_grid_iterator.output is None
    assert not my_grid_iterator.num_grid_points_per_axis
    assert my_grid_iterator.num_parameters == num_parameters
    assert not my_grid_iterator.scale_type


def test_model_evaluate(default_grid_iterator, mocker):
    """TODO_doc."""
    mp = mocker.patch('queens.models.simulation_model.SimulationModel.evaluate', return_value=None)
    default_grid_iterator.model.evaluate(None)
    mp.assert_called_once()


def test_pre_run_one(
    grid_dict_one,
    parameters_one,
    expected_samples_one,
    result_description,
    default_model,
    _initialize_global_settings,
):
    """TODO_doc."""
    grid_iterator = GridIterator(
        model=default_model,
        parameters=parameters_one,
        result_description=result_description,
        grid_design=grid_dict_one,
    )
    grid_iterator.pre_run()
    np.testing.assert_array_equal(grid_iterator.samples, expected_samples_one)


def test_pre_run_two(
    grid_dict_two,
    parameters_two,
    result_description,
    expected_samples_two,
    default_model,
    _initialize_global_settings,
):
    """TODO_doc."""
    grid_iterator = GridIterator(
        model=default_model,
        parameters=parameters_two,
        result_description={},
        grid_design=grid_dict_two,
    )
    grid_iterator.pre_run()
    np.testing.assert_array_equal(grid_iterator.samples, expected_samples_two)


def test_pre_run_three(
    grid_dict_three,
    parameters_three,
    expected_samples_three,
    result_description,
    default_model,
    _initialize_global_settings,
):
    """TODO_doc."""
    grid_iterator = GridIterator(
        model=default_model,
        parameters=parameters_three,
        result_description=result_description,
        grid_design=grid_dict_three,
    )
    grid_iterator.pre_run()
    np.testing.assert_array_equal(grid_iterator.samples, expected_samples_three)


def test_core_run(mocker, default_grid_iterator, expected_samples_two):
    """TODO_doc."""
    mocker.patch('queens.models.simulation_model.SimulationModel.evaluate', return_value=2)
    default_grid_iterator.samples = expected_samples_two
    default_grid_iterator.core_run()
    np.testing.assert_array_equal(default_grid_iterator.samples, expected_samples_two)
    assert default_grid_iterator.output == 2


def test_post_run(mocker, default_grid_iterator):
    """TODO_doc."""
    # test if save results is called
    mp1 = mocker.patch('queens.iterators.grid_iterator.write_results', return_value=None)
    mocker.patch(
        'queens.visualization.grid_iterator_visualization.grid_iterator_visualization_instance',
    )
    mp3 = mocker.patch(
        'queens.visualization.grid_iterator_visualization.grid_iterator_visualization_instance'
        '.plot_QoI_grid',
        return_value=1,
    )
    default_grid_iterator.post_run()
    mp1.assert_called_once()
    mp3.assert_called_once()
