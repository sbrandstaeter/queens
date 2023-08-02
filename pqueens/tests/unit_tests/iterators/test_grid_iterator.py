"""TODO_doc."""

import numpy as np
import pytest
from mock import Mock

import pqueens.parameters.parameters as parameters_module
from pqueens.iterators.grid_iterator import GridIterator
from pqueens.models.simulation_model import SimulationModel


# general input fixtures
@pytest.fixture()
def global_settings():
    """TODO_doc."""
    global_set = {'output_dir': 'dummyoutput', 'experiment_name': 'dummy_exp_name'}
    return global_set


@pytest.fixture()
def grid_dict_one():
    """TODO_doc."""
    axis_description = {"num_grid_points": 5, "axis_type": "lin", "data_type": "FLOAT"}
    grid_dict_dummy = {"x1": axis_description}
    return grid_dict_dummy


@pytest.fixture()
def grid_dict_two():
    """TODO_doc."""
    axis_description = {"num_grid_points": 5, "axis_type": "lin", "data_type": "FLOAT"}
    grid_dict_dummy = {"x1": axis_description, "x2": axis_description}
    return grid_dict_dummy


@pytest.fixture()
def grid_dict_three():
    """TODO_doc."""
    axis_description = {"num_grid_points": 5, "axis_type": "lin", "data_type": "FLOAT"}
    grid_dict_dummy = {"x1": axis_description, "x2": axis_description, "x3": axis_description}
    return grid_dict_dummy


@pytest.fixture()
def grid_dict_four():
    """TODO_doc."""
    axis_description = {"num_grid_points": 5, "axis_type": "lin", "data_type": "FLOAT"}
    grid_dict_dummy = {
        "x1": axis_description,
        "x2": axis_description,
        "x3": axis_description,
        "x4": axis_description,
    }
    return grid_dict_dummy


def parameters_one():
    """TODO_doc."""
    rv = {"type": "uniform", "lower_bound": -2, "upper_bound": 2}
    params = {"x1": rv}
    parameters_module.from_config_create_parameters({"parameters": params})
    return params


@pytest.fixture()
def parameters_two():
    """TODO_doc."""
    rv = {"type": "uniform", "lower_bound": -2, "upper_bound": 2}
    params = {"x1": rv, "x2": rv}
    parameters_module.from_config_create_parameters({"parameters": params})
    return params


def parameters_three():
    """TODO_doc."""
    rv = {"type": "uniform", "lower_bound": -2, "upper_bound": 2}
    params = {"x1": rv, "x2": rv, "x3": rv}
    parameters_module.from_config_create_parameters({"parameters": params})
    return params


@pytest.fixture()
def parameters_four():
    """TODO_doc."""
    rv = {"type": "uniform", "lower_bound": -2, "upper_bound": 2}
    params = {"x1": rv, "x2": rv, "x3": rv, "x4": rv}
    parameters_module.from_config_create_parameters({"parameters": params})
    return params


@pytest.fixture()
def result_description():
    """TODO_doc."""
    description = {"write_results": True}
    return description


@pytest.fixture()
def expected_samples_one():
    """TODO_doc."""
    x1 = np.linspace(-2, 2, 5)
    return np.atleast_2d(x1).T


@pytest.fixture()
def expected_samples_two():
    """TODO_doc."""
    x1 = np.linspace(-2, 2, 5)
    x2 = np.linspace(-2, 2, 5)
    X1, X2 = np.meshgrid(x1, x2)
    samples = np.array([X1.flatten(), X2.flatten()]).T
    return samples


@pytest.fixture()
def expected_samples_three():
    """TODO_doc."""
    x1 = np.linspace(-2, 2, 5)
    x2 = np.linspace(-2, 2, 5)
    x3 = np.linspace(-2, 2, 5)
    X1, X2, X3 = np.meshgrid(x1, x2, x3)
    samples = np.array([X1.flatten(), X2.flatten(), X3.flatten()]).T
    return samples


# fixtures for some objects
@pytest.fixture()
def default_model(parameters_two):
    """TODO_doc."""
    interface = 'dummy_interface'
    model = SimulationModel(interface)
    return model


@pytest.fixture()
def default_grid_iterator(
    global_settings, grid_dict_two, parameters_two, default_model, result_description
):
    """TODO_doc."""
    # create iterator object
    my_grid_iterator = GridIterator(
        model=default_model,
        result_description=result_description,
        global_settings=global_settings,
        grid_design=grid_dict_two,
    )
    return my_grid_iterator


# -------------- actual unit_tests --------------------------------------------------
def test_init(
    mocker, global_settings, grid_dict_two, parameters_two, default_model, result_description
):
    """TODO_doc."""
    # some default input for testing
    num_parameters = 2
    mp = mocker.patch('pqueens.iterators.iterator.Iterator.__init__')
    GridIterator.parameters = Mock()
    GridIterator.parameters.num_parameters = num_parameters

    my_grid_iterator = GridIterator(
        model=default_model,
        result_description=result_description,
        grid_design=grid_dict_two,
        global_settings=global_settings,
    )

    # tests / asserts
    mp.assert_called_once_with(default_model, global_settings)
    assert my_grid_iterator.grid_dict == grid_dict_two
    assert my_grid_iterator.result_description == result_description
    assert my_grid_iterator.samples is None
    assert my_grid_iterator.output is None
    assert not my_grid_iterator.num_grid_points_per_axis
    assert my_grid_iterator.num_parameters == num_parameters
    assert not my_grid_iterator.scale_type


def test_model_evaluate(default_grid_iterator, mocker):
    """TODO_doc."""
    mp = mocker.patch('pqueens.models.simulation_model.SimulationModel.evaluate', return_value=None)
    default_grid_iterator.model.evaluate(None)
    mp.assert_called_once()


def test_pre_run_one(
    grid_dict_one,
    expected_samples_one,
    result_description,
    default_model,
    global_settings,
):
    """TODO_doc."""
    parameters_one()
    grid_iterator = GridIterator(
        model=default_model,
        result_description=result_description,
        global_settings=global_settings,
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
    global_settings,
):
    """TODO_doc."""
    grid_iterator = GridIterator(
        model=default_model,
        result_description={},
        global_settings=global_settings,
        grid_design=grid_dict_two,
    )
    grid_iterator.pre_run()
    np.testing.assert_array_equal(grid_iterator.samples, expected_samples_two)


def test_pre_run_three(
    grid_dict_three,
    expected_samples_three,
    result_description,
    default_model,
    global_settings,
):
    """TODO_doc."""
    num_params = 3
    parameters_three()
    grid_iterator = GridIterator(
        model=default_model,
        result_description=result_description,
        global_settings=global_settings,
        grid_design=grid_dict_three,
    )
    grid_iterator.pre_run()
    np.testing.assert_array_equal(grid_iterator.samples, expected_samples_three)


def test_core_run(mocker, default_grid_iterator, expected_samples_two):
    """TODO_doc."""
    mocker.patch('pqueens.models.simulation_model.SimulationModel.evaluate', return_value=2)
    default_grid_iterator.samples = expected_samples_two
    default_grid_iterator.core_run()
    np.testing.assert_array_equal(default_grid_iterator.samples, expected_samples_two)
    assert default_grid_iterator.output == 2


def test_post_run(mocker, default_grid_iterator, mock_visualization):
    """TODO_doc."""
    # test if save results is called
    mp1 = mocker.patch('pqueens.iterators.grid_iterator.write_results', return_value=None)
    mocker.patch(
        'pqueens.visualization.grid_iterator_visualization.grid_iterator_visualization_instance',
    )
    mp3 = mocker.patch(
        'pqueens.visualization.grid_iterator_visualization.grid_iterator_visualization_instance'
        '.plot_QoI_grid',
        return_value=1,
    )
    default_grid_iterator.post_run()
    mp1.assert_called_once()
    mp3.assert_called_once()
