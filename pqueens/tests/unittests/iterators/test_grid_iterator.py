import numpy as np
import pytest
from pqueens.iterators.grid_iterator import GridIterator
from pqueens.models.simulation_model import SimulationModel


# general input fixtures
@pytest.fixture()
def global_settings():
    global_set = {'output_dir': 'dummyoutput', 'experiment_name': 'dummy_exp_name'}
    return global_set


@pytest.fixture()
def grid_dict_one():
    grid_dict_dummy = {
        "x1": {"num_grid_points": 5, "axis_type": "lin"},
    }
    return grid_dict_dummy


@pytest.fixture()
def grid_dict_two():
    grid_dict_dummy = {
        "x1": {"num_grid_points": 5, "axis_type": "lin"},
        "x2": {"num_grid_points": 5, "axis_type": "lin"},
    }
    return grid_dict_dummy


@pytest.fixture()
def grid_dict_three():
    grid_dict_dummy = {
        "x1": {"num_grid_points": 5, "axis_type": "lin"},
        "x2": {"num_grid_points": 5, "axis_type": "lin"},
        "x3": {"num_grid_points": 5, "axis_type": "lin"},
    }
    return grid_dict_dummy


@pytest.fixture()
def grid_dict_four():
    grid_dict_dummy = {
        "x1": {"num_grid_points": 5, "axis_type": "lin"},
        "x2": {"num_grid_points": 5, "axis_type": "lin"},
        "x3": {"num_grid_points": 5, "axis_type": "lin"},
        "x4": {"num_grid_points": 5, "axis_type": "lin"},
    }
    return grid_dict_dummy


@pytest.fixture()
def parameters_one():
    params = {"random_variables": {"x1": {"type": "FLOAT", "size": 1, "min": -2.0, "max": 2.0}}}
    return params


@pytest.fixture()
def parameters_two():
    params = {
        "random_variables": {
            "x1": {"type": "FLOAT", "size": 1, "min": -2.0, "max": 2.0},
            "x2": {"type": "FLOAT", "size": 1, "min": -2.0, "max": 2.0},
        }
    }
    return params


@pytest.fixture()
def parameters_three():
    params = {
        "random_variables": {
            "x1": {"type": "FLOAT", "size": 1, "min": -2.0, "max": 2.0},
            "x2": {"type": "FLOAT", "size": 1, "min": -2.0, "max": 2.0},
            "x3": {"type": "FLOAT", "size": 1, "min": -2.0, "max": 2.0},
        }
    }
    return params


@pytest.fixture()
def parameters_four():
    params = {
        "random_variables": {
            "x1": {"type": "FLOAT", "size": 1, "min": -2.0, "max": 2.0},
            "x2": {"type": "FLOAT", "size": 1, "min": -2.0, "max": 2.0},
            "x3": {"type": "FLOAT", "size": 1, "min": -2.0, "max": 2.0},
            "x4": {"type": "FLOAT", "size": 1, "min": -2.0, "max": 2.0},
        }
    }
    return params


@pytest.fixture()
def result_description():
    description = {"write_results": True}
    return description


@pytest.fixture()
def expected_samples_one():
    x1 = np.linspace(-2, 2, 5)
    return np.atleast_2d(x1).T


@pytest.fixture()
def expected_samples_two():
    x1 = np.linspace(-2, 2, 5)
    x2 = np.linspace(-2, 2, 5)
    X1, X2 = np.meshgrid(x1, x2)
    samples = np.array([X1.flatten(), X2.flatten()]).T
    return samples


@pytest.fixture()
def expected_samples_three():
    x1 = np.linspace(-2, 2, 5)
    x2 = np.linspace(-2, 2, 5)
    x3 = np.linspace(-2, 2, 5)
    X1, X2, X3 = np.meshgrid(x1, x2, x3)
    samples = np.array([X1.flatten(), X2.flatten(), X3.flatten()]).T
    return samples


# fixtures for some objects
@pytest.fixture()
def default_model(parameters_two):
    model_name = 'dummy_model'
    interface = 'dummy_interface'
    model = SimulationModel(model_name, interface, parameters_two)
    return model


@pytest.fixture()
def default_grid_iterator(
    global_settings, grid_dict_two, parameters_two, default_model, result_description
):

    # create iterator object
    num_parameters = 2
    my_grid_iterator = GridIterator(
        default_model,
        result_description,
        global_settings,
        grid_dict_two,
        parameters_two,
        num_parameters,
    )
    return my_grid_iterator


# -------------- actual unittests --------------------------------------------------
def test_init(
    mocker, global_settings, grid_dict_two, parameters_two, default_model, result_description
):

    # some default input for testing
    num_parameters = 2
    mp = mocker.patch('pqueens.iterators.iterator.Iterator.__init__')

    my_grid_iterator = GridIterator(
        default_model,
        result_description,
        global_settings,
        grid_dict_two,
        parameters_two,
        num_parameters,
    )

    # tests / asserts
    mp.assert_called_once_with(default_model, global_settings)
    assert my_grid_iterator.grid_dict == grid_dict_two
    assert my_grid_iterator.parameters == parameters_two
    assert my_grid_iterator.result_description == result_description
    assert my_grid_iterator.samples is None
    assert my_grid_iterator.output is None
    assert my_grid_iterator.num_grid_points_per_axis == []
    assert my_grid_iterator.num_parameters == num_parameters
    assert my_grid_iterator.scale_type == []


def test_eval_model(default_grid_iterator, mocker):
    mp = mocker.patch('pqueens.models.simulation_model.SimulationModel.evaluate', return_value=None)
    default_grid_iterator.eval_model()
    mp.assert_called_once()


def test_pre_run_one(
    grid_dict_one,
    parameters_one,
    expected_samples_one,
    result_description,
    default_model,
    global_settings,
):
    num_params = 1
    grid_iterator = GridIterator(
        default_model,
        result_description,
        global_settings,
        grid_dict_one,
        parameters_one["random_variables"],
        num_params,
    )
    grid_iterator.pre_run()
    np.testing.assert_array_equal(grid_iterator.samples, expected_samples_one)


def test_pre_run_two(
    grid_dict_two, parameters_two, expected_samples_two, default_model, global_settings
):
    num_params = 2
    grid_iterator = GridIterator(
        default_model,
        result_description,
        global_settings,
        grid_dict_two,
        parameters_two["random_variables"],
        num_params,
    )
    grid_iterator.pre_run()
    np.testing.assert_array_equal(grid_iterator.samples, expected_samples_two)


def test_pre_run_three(
    grid_dict_three,
    parameters_three,
    expected_samples_three,
    result_description,
    default_model,
    global_settings,
):
    num_params = 3
    grid_iterator = GridIterator(
        default_model,
        result_description,
        global_settings,
        grid_dict_three,
        parameters_three["random_variables"],
        num_params,
    )
    grid_iterator.pre_run()
    np.testing.assert_array_equal(grid_iterator.samples, expected_samples_three)


def test_pre_run_four(
    grid_dict_four, parameters_four, result_description, default_model, global_settings
):
    num_params = 4
    grid_iterator = GridIterator(
        default_model,
        result_description,
        global_settings,
        grid_dict_four,
        parameters_four["random_variables"],
        num_params,
    )
    with pytest.raises(ValueError) as e:
        grid_iterator.pre_run()
    assert str(e.value) == "More than 3 grid parameters are currently not supported! Abort..."


def test_core_run(mocker, default_grid_iterator, expected_samples_two):
    mocker.patch('pqueens.iterators.grid_iterator.GridIterator.eval_model', return_value=2)
    default_grid_iterator.samples = expected_samples_two
    default_grid_iterator.core_run()
    variable_list = default_grid_iterator.model.variables
    var_vec = np.array([var.get_active_variables_vector() for var in variable_list]).reshape(-1, 2)
    np.testing.assert_array_equal(var_vec, expected_samples_two)
    assert default_grid_iterator.output == 2


# custom class to mock the visualization module
class InstanceMock:
    @staticmethod
    def plot_QoI_grid(self, *args, **kwargs):
        return 1


@pytest.fixture
def mock_visualization():
    my_mock = InstanceMock()
    return my_mock


def test_post_run(mocker, default_grid_iterator, mock_visualization):
    # test if save results is called
    mp1 = mocker.patch('pqueens.iterators.grid_iterator.write_results', return_value=None)
    mocker.patch(
        'pqueens.visualization.grid_iterator_visualization.grid_iterator_visualization_instance',
        return_value=mock_visualization,
    )
    mp3 = mocker.patch(
        'pqueens.visualization.grid_iterator_visualization.grid_iterator_visualization_instance'
        '.plot_QoI_grid',
        return_value=1,
    )
    default_grid_iterator.post_run()
    mp1.assert_called_once()
    mp3.assert_called_once()
