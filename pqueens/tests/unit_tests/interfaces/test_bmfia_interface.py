"""Unit tests for the Bayesian multi-fidelity inverse analysis interface."""

import time

import numpy as np
import pytest

from pqueens.interfaces.bmfia_interface import BmfiaInterface
from pqueens.utils.exceptions import InvalidOptionError


# ---- Fixtures and helper methods / classes ---------
@pytest.fixture
def default_bmfia_interface():
    """Fixture for a dummy bmfia interface."""
    num_processors_multi_processing = 2
    instantiate_probabilistic_mappings = BmfiaInterface._instantiate_per_coordinate
    evaluate_method = BmfiaInterface._evaluate_per_coordinate
    evaluate_and_gradient_method = BmfiaInterface._evaluate_and_gradient_per_coordinate
    update_mappings_method = BmfiaInterface._update_mappings_per_coordinate
    default_interface = BmfiaInterface(
        instantiate_probabilistic_mappings,
        num_processors_multi_processing,
        evaluate_method,
        evaluate_and_gradient_method,
        update_mappings_method,
    )
    return default_interface


class DummyRegression:
    """A dummy regression class."""

    def __init__(self):
        """Init for dummy regression."""
        self.state = 0

    def predict(self, *_, **__):
        """A dummy predict method."""
        return {"mean": np.array([1, 2]), "variance": np.array([4, 5])}

    def train(self, *_, **__):
        """A dummy training method."""
        time.sleep(0.01)

    def set_state(self, *_, **__):
        """A dummy *set_state* method."""
        self.state = 1

    def get_state(self, *_, **__):
        """A dummy *get_state* method."""
        return {'test': 'test'}


@pytest.fixture
def dummy_reg_obj():
    """Fixture for a dummy regression object."""
    obj = DummyRegression()
    return obj


@pytest.fixture
def default_probabilistic_obj_lst(dummy_reg_obj):
    """Fixture for probabilistic mapping objects."""
    dummy_lst = [dummy_reg_obj] * 3
    return dummy_lst


@pytest.fixture
def my_state_lst():
    """Fixture for a dummy state list."""
    return [1, 2, 3]


class MyContextManagerPool:
    """A dummy context manager pool class."""

    def __init__(self, *_, **__):
        """Init context manager pool."""
        self.dummy = 0

    def __enter__(self):
        """Dummy enter method for context manager."""
        return self

    def __exit__(self, *_, **__):
        """Dummy exit method for context manager."""
        pass

    def imap(self, *_, **__):
        """A dummy map method for the dummy pool."""
        return [1, 2, 3]

    def close(self):
        """A dummy close method."""
        pass


class MyContext:
    """Dummy context class."""

    Pool = MyContextManagerPool

    def __init__(self, *_, **__):
        """Init method for dummy class."""
        self.dummy = 0


@pytest.fixture()
def dummy_plot_instance():
    """Quick fake plotting object."""
    my_plot_obj = my_plot()

    class my_plot:
        """Fake plot class."""

        def plot():
            """Fake plot method."""
            pass

    return my_plot_obj


# ---- Actual unit_tests ------------------------------
def test_fcc():
    """Test from config create method."""
    interface_name = "bmfia_interface"

    # test configuration with settings per_coordinate
    dummy_config = {
        "bmfia_interface": {
            "num_processors_multi_processing": 2,
            "regression_strategy": "per_coordinate",
        }
    }
    bmfia_interface = BmfiaInterface.from_config_create_interface(interface_name, dummy_config)

    assert (
        bmfia_interface.instantiate_probabilistic_mappings.__func__
        is BmfiaInterface._instantiate_per_coordinate
    )
    assert bmfia_interface.num_processors_multi_processing == 2
    assert bmfia_interface.probabilistic_mapping_obj_lst == []
    assert bmfia_interface.evaluate_method.__func__ is BmfiaInterface._evaluate_per_coordinate
    assert (
        bmfia_interface.evaluate_and_gradient_method.__func__
        is BmfiaInterface._evaluate_and_gradient_per_coordinate
    )
    assert (
        bmfia_interface.update_mappings_method.__func__
        is BmfiaInterface._update_mappings_per_coordinate
    )
    assert bmfia_interface.coord_labels is None
    assert bmfia_interface.time_vec is None
    assert bmfia_interface.coords_mat is None

    # test configuration with settings per_time_step
    dummy_config["bmfia_interface"]["probabilistic_mapping_type"] = "per_time_step"
    bmfia_interface = BmfiaInterface.from_config_create_interface(interface_name, dummy_config)

    assert (
        bmfia_interface.instantiate_probabilistic_mappings.__func__
        is BmfiaInterface._instantiate_per_time_step
    )
    assert bmfia_interface.num_processors_multi_processing == 2
    assert bmfia_interface.probabilistic_mapping_obj_lst == []
    assert bmfia_interface.evaluate_method.__func__ is BmfiaInterface._evaluate_per_time_step
    assert (
        bmfia_interface.evaluate_and_gradient_method.__func__
        is BmfiaInterface._evaluate_and_gradient_per_time_step
    )
    assert (
        bmfia_interface.update_mappings_method.__func__
        is BmfiaInterface._update_mappings_per_time_step
    )
    assert bmfia_interface.coord_labels is None
    assert bmfia_interface.time_vec is None
    assert bmfia_interface.coords_mat is None

    # test wrong configuration
    dummy_config["bmfia_interface"]["probabilistic_mapping_type"] = "blabla"
    with pytest.raises(InvalidOptionError):
        BmfiaInterface.from_config_create_interface(interface_name, dummy_config)


def test__init__():
    """Test the instantiation of the interface object."""
    instantiate_probabilistic_mappings = BmfiaInterface._instantiate_per_coordinate
    num_processors_multi_processing = 3
    evaluate_method = BmfiaInterface._evaluate_per_coordinate
    evaluate_and_gradient_method = BmfiaInterface._evaluate_and_gradient_per_coordinate
    update_mappings_method = BmfiaInterface._update_mappings_per_coordinate

    interface = BmfiaInterface(
        instantiate_probabilistic_mappings,
        num_processors_multi_processing,
        evaluate_method,
        evaluate_and_gradient_method,
        update_mappings_method,
    )

    assert interface.instantiate_probabilistic_mappings == instantiate_probabilistic_mappings
    assert interface.num_processors_multi_processing == num_processors_multi_processing
    assert interface.probabilistic_mapping_obj_lst == []
    assert interface.evaluate_method == evaluate_method
    assert interface.evaluate_and_gradient_method == evaluate_and_gradient_method
    assert interface.update_mappings_method == update_mappings_method
    assert interface.coord_labels is None
    assert interface.time_vec is None
    assert interface.coords_mat is None


def test_build_approximation(default_bmfia_interface, mocker, default_probabilistic_obj_lst):
    """Test the set-up / build of the probabilistic regression models."""
    z_lf_train = np.zeros((2, 30))
    y_hf_train = np.zeros((2, 25))
    dummy_lst = [1, 2, 3]
    config = {"dummy": "dummy"}
    coord_labels = ["x1", "x2"]
    time_vec = None
    coords_mat = np.array([[0, 1], [0, 1]])
    approx_name = "bmfia"

    # pylint: disable=line-too-long
    mock_train_parallel = mocker.patch(
        'pqueens.interfaces.bmfia_interface.BmfiaInterface._train_probabilistic_mappings_in_parallel',
        return_value=dummy_lst,
    )
    mock_optimize_state = mocker.patch(
        'pqueens.interfaces.bmfia_interface.BmfiaInterface._set_optimized_state_of_probabilistic_mappings'
    )
    mocker.patch(
        'pqueens.visualization.bmfia_visualization.bmfia_visualization_instance',
        return_value=dummy_plot_instance,
    )

    # pylint: disable=line-too-long

    default_bmfia_interface.num_processors_multi_processing = 3

    # Test with wrong input dimensions (2D Tensor) --> ValueError
    with pytest.raises(IndexError):
        default_bmfia_interface.build_approximation(
            z_lf_train, y_hf_train, config, approx_name, coord_labels, time_vec, coords_mat
        )

    # Test with wrong input dimensions --> AssertionError
    z_lf_train = np.zeros((1, 2, 30))
    with pytest.raises(IndexError):
        default_bmfia_interface.build_approximation(
            z_lf_train, y_hf_train, config, approx_name, coord_labels, time_vec, coords_mat
        )

    # Test with correct input dimensions for Z_LF_train and Y_HF_train
    y_hf_train = np.zeros((2, 30))
    mock_instantiate = mocker.MagicMock()
    mock_instantiate.return_value = (z_lf_train, y_hf_train, default_probabilistic_obj_lst)
    default_bmfia_interface.instantiate_probabilistic_mappings = mock_instantiate

    num_coords = z_lf_train.T.shape[2]
    default_bmfia_interface.build_approximation(
        z_lf_train, y_hf_train, config, approx_name, coord_labels, time_vec, coords_mat
    )

    # -- Actual assert / tests ---
    assert mock_instantiate.call_once()
    assert mock_train_parallel.call_once()
    assert mock_optimize_state.call_once()

    np.testing.assert_array_equal(mock_instantiate.call_args[0][0], z_lf_train)
    np.testing.assert_array_equal(mock_instantiate.call_args[0][1], y_hf_train)
    np.testing.assert_array_equal(mock_train_parallel.call_args[0][0], num_coords)
    np.testing.assert_array_equal(mock_optimize_state.call_args[0][0], dummy_lst)


def test_instantiate_per_coordinate(
    default_bmfia_interface, mocker, dummy_reg_obj, default_probabilistic_obj_lst
):
    """Test the instantiation of the probabilistic mappings."""
    z_lf_train = np.zeros((1, 2))
    y_hf_train = np.zeros((1, 2, 3))
    num_reg = y_hf_train.shape[2]
    default_bmfia_interface.probabilistic_mapping_obj_lst = []
    time_vec = None
    coords_mat = np.array([[0, 1], [0, 1]])
    approx_name = "bmfia"
    config = {"dummy": "dummy"}

    # test wrong z_lf_train input dimensions
    with pytest.raises(IndexError):
        default_bmfia_interface._instantiate_per_coordinate(
            z_lf_train, y_hf_train, time_vec, coords_mat, config, approx_name
        )

    # test correct z_lf_train input dimensions
    z_lf_train = np.zeros((1, 2, 3))
    mp_1 = mocker.patch(
        'pqueens.interfaces.bmfia_interface.from_config_create_regression_approximation',
        return_value=dummy_reg_obj,
    )
    (
        z_lf_array_out,
        y_hf_array_out,
        probabilistic_mapping_ob_lst,
    ) = default_bmfia_interface._instantiate_per_coordinate(
        z_lf_train, y_hf_train, time_vec, coords_mat, config, approx_name
    )

    # --- asserts / tests
    assert default_probabilistic_obj_lst == probabilistic_mapping_ob_lst
    assert mp_1.call_count == num_reg
    assert mp_1.call_args[0][0] == config
    assert mp_1.call_args[0][1] == approx_name
    np.testing.assert_array_equal(z_lf_array_out, z_lf_train)
    np.testing.assert_array_equal(y_hf_array_out, y_hf_train)


def test_instantiate_per_time_step(mocker):
    """Test the instantiation of the probabilistic mappings."""
    z_lf_train = np.zeros((1, 2, 3))
    y_hf_train = np.zeros((2, 2, 2))
    z_lf_array_out = np.zeros((2, 1, 3))
    num_coords = 3
    t_size = 2
    time_vec = np.array([0, 1])
    coords_mat = np.array([[0, 1], [0, 1]])
    config = {"dummy": "dummy"}
    approx_name = "bmfia"

    mp_1 = mocker.patch(
        "pqueens.interfaces.bmfia_interface.BmfiaInterface._check_coordinates_return_dimensions",
        return_value=(num_coords, t_size),
    )
    mp_2 = mocker.patch(
        "pqueens.interfaces.bmfia_interface.BmfiaInterface._prepare_z_lf_for_time_steps",
        return_value=z_lf_array_out,
    )
    mp_3 = mocker.patch(
        "pqueens.interfaces.bmfia_interface.from_config_create_regression_approximation",
        return_value=dummy_reg_obj,
    )

    (
        z_lf_array,
        y_hf_array,
        probabilistic_mapping_obj_lst,
    ) = BmfiaInterface._instantiate_per_time_step(
        z_lf_train, y_hf_train, time_vec, coords_mat, config, approx_name
    )

    # --- asserts / tests
    mp_1.assert_called_once()
    mp_1.assert_called_with(z_lf_train, time_vec, coords_mat)

    mp_2.assert_called_once()
    mp_2.assert_called_with(z_lf_train, t_size, coords_mat)

    assert mp_3.call_count == 2
    np.testing.assert_array_equal(z_lf_array, z_lf_array_out)
    np.testing.assert_array_equal(y_hf_array, np.zeros((2, 4, 1)))
    assert probabilistic_mapping_obj_lst == [dummy_reg_obj, dummy_reg_obj]


def test_train_probabilistic_mappings_in_parallel(
    default_bmfia_interface, mocker, my_state_lst, default_probabilistic_obj_lst
):
    """Test the parallel training of the mappings."""
    Z_LF_train = np.zeros((1, 2, 3))
    default_bmfia_interface.probabilistic_mapping_obj_lst = default_probabilistic_obj_lst
    mocker.patch('pqueens.interfaces.bmfia_interface.get_context', MyContext)

    # test with valid configuration
    num_coords = Z_LF_train.T.shape[0]
    num_processors_multi_processing = 3
    return_state_list = BmfiaInterface._train_probabilistic_mappings_in_parallel(
        num_coords, num_processors_multi_processing, default_probabilistic_obj_lst
    )
    # --- asserts / tests ---
    assert my_state_lst == return_state_list

    # test with no specification for processors
    num_processors_multi_processing = None
    with pytest.raises(RuntimeError):
        BmfiaInterface._train_probabilistic_mappings_in_parallel(
            num_coords, num_processors_multi_processing, default_probabilistic_obj_lst
        )

    # test with 0 as a specification for processors
    num_processors_multi_processing = 0
    with pytest.raises(RuntimeError):
        BmfiaInterface._train_probabilistic_mappings_in_parallel(
            num_coords, num_processors_multi_processing, default_probabilistic_obj_lst
        )

    # test with float as a specification for processors
    default_bmfia_interface.num_processors_multi_processing = 1.3
    with pytest.raises(RuntimeError):
        BmfiaInterface._train_probabilistic_mappings_in_parallel(
            num_coords, num_processors_multi_processing, default_probabilistic_obj_lst
        )

    # test with str as a specification for processors
    default_bmfia_interface.num_processors_multi_processing = 'blabla'
    with pytest.raises(RuntimeError):
        BmfiaInterface._train_probabilistic_mappings_in_parallel(
            num_coords, num_processors_multi_processing, default_probabilistic_obj_lst
        )

    # test with too large number of processors
    default_bmfia_interface.num_processors_multi_processing = 99999999
    with pytest.raises(RuntimeError):
        default_bmfia_interface._train_probabilistic_mappings_in_parallel(
            num_coords, num_processors_multi_processing, default_probabilistic_obj_lst
        )


def test_set_optimized_state_of_probabilistic_mappings(
    default_bmfia_interface, my_state_lst, mocker, default_probabilistic_obj_lst
):
    """Test the state update of the mappings."""
    mocker.patch(
        'pqueens.interfaces.bmfia_interface.from_config_create_regression_approximation',
        DummyRegression,
    )
    default_bmfia_interface.probabilistic_mapping_obj_lst = default_probabilistic_obj_lst
    default_bmfia_interface._set_optimized_state_of_probabilistic_mappings(my_state_lst)
    for obj in default_bmfia_interface.probabilistic_mapping_obj_lst:
        assert obj.state == 1


def test_optimize_hyper_params(mocker, dummy_reg_obj):
    """Test the training of a single mapping."""
    mo_1 = mocker.patch(__name__ + '.DummyRegression.train')
    state_dict = BmfiaInterface._optimize_hyper_params(dummy_reg_obj)

    # asserts / tests
    mo_1.assert_called_once()
    assert state_dict == {'test': 'test'}


def test_evaluate(default_bmfia_interface, mocker):
    """Test the evaluation of the interface."""
    # general inputs
    default_bmfia_interface.probabilistic_mapping_obj_lst = ["dummy", "dummy"]
    Z_LF = np.array([[1, 2]])
    support = "y"

    per_coordinate_return = (np.array([[1, 2]]), np.array([[3, 4]]))
    mp1 = mocker.MagicMock()
    mp1.return_value = per_coordinate_return
    default_bmfia_interface.evaluate_method = mp1

    mean, var = default_bmfia_interface.evaluate(Z_LF, support)

    mp1.assert_called_once()
    mp1.assert_called_with(
        Z_LF,
        support,
        ['dummy', 'dummy'],
        default_bmfia_interface.time_vec,
        default_bmfia_interface.coords_mat,
    )
    np.testing.assert_almost_equal(mean, per_coordinate_return[0])
    np.testing.assert_almost_equal(var, per_coordinate_return[1])


def test_evaluate_and_gradient(default_bmfia_interface, mocker):
    """Test the evaluation and gradient of the interface."""
    # general inputs
    default_bmfia_interface.probabilistic_mapping_obj_lst = ["dummy", "dummy"]
    Z_LF = np.array([[1, 2]])
    support = "y"

    # test evaluation per coordinate
    per_coordinate_return = (
        np.array([[1, 2]]),
        np.array([[3, 4]]),
        np.array([[4, 5], [6, 7]]),
        np.array([[8, 9], [10, 11]]),
    )
    mp1 = mocker.MagicMock()
    mp1.return_value = per_coordinate_return
    default_bmfia_interface.evaluate_and_gradient_method = mp1

    mean, var, grad_mean, grad_var = default_bmfia_interface.evaluate_and_gradient(Z_LF, support)

    mp1.assert_called_once()
    mp1.assert_called_with(
        Z_LF,
        support,
        ['dummy', 'dummy'],
        default_bmfia_interface.time_vec,
        default_bmfia_interface.coords_mat,
    )
    np.testing.assert_almost_equal(mean, per_coordinate_return[0])
    np.testing.assert_almost_equal(var, per_coordinate_return[1])
    np.testing.assert_almost_equal(grad_mean, per_coordinate_return[2])
    np.testing.assert_almost_equal(grad_var, per_coordinate_return[3])


def test_evaluate_per_coordinate(default_bmfia_interface, mocker, default_probabilistic_obj_lst):
    """Test the evaluation per coordinate."""
    # general inputs
    support = "y"
    z_lf = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    # test evaluation per coordinate with correct
    map_1 = mocker.MagicMock()
    map_1.predict.return_value = {"mean": np.array([[1], [2]]), "variance": np.array([[3], [4]])}

    map_2 = mocker.MagicMock()
    map_2.predict.return_value = {"mean": np.array([[3], [4]]), "variance": np.array([[5], [6]])}

    probabilistic_mapping_obj_lst = [map_1, map_2]
    mean, variance = default_bmfia_interface._evaluate_per_coordinate(
        z_lf, support, probabilistic_mapping_obj_lst, None, None
    )

    assert map_1.predict.called_once()
    assert map_2.predict.called_once()

    np.testing.assert_array_equal(map_1.predict.call_args[0][0], z_lf.T[0, :, :])
    assert map_1.predict.call_args[1]["support"] == support
    assert map_1.predict.call_args[1]["gradient_bool"] is False

    np.testing.assert_array_equal(map_2.predict.call_args[0][0], z_lf.T[1, :, :])
    assert map_2.predict.call_args[1]["support"] == support
    assert map_2.predict.call_args[1]["gradient_bool"] is False

    np.testing.assert_array_equal(mean, np.array([[1, 3], [2, 4]]))
    np.testing.assert_array_equal(variance, np.array([[3, 5], [4, 6]]))


def test_evaluate_per_time_step(default_bmfia_interface, mocker):
    """Test the evaluation per time step."""
    # general inputs
    z_lf = np.array([[[1, 2], [3, 4]]])
    z_lf_array = np.array([[[1, 2, 3, 4]], [[5, 6, 7, 8]]])
    support = "y"
    default_mean = np.array([[7], [8]])
    default_variance = np.array([[11], [12]])
    time_vec = np.array([1, 2])  # two time steps
    coords_mat = np.array([1, 2])  # one coordinates in 2d
    num_coords = 1

    # mock check coordinate compliance
    mp1 = mocker.patch(
        "pqueens.interfaces.bmfia_interface.BmfiaInterface._check_coordinates_return_dimensions",
        return_value=(1, 2),
    )
    mp2 = mocker.patch(
        "pqueens.interfaces.bmfia_interface.BmfiaInterface._prepare_z_lf_for_time_steps",
        return_value=z_lf_array,
    )
    mp3 = mocker.patch(
        "pqueens.interfaces.bmfia_interface.BmfiaInterface._iterate_over_time_steps",
        return_value=(default_mean, default_variance, None, None),
    )

    mean, variance = BmfiaInterface._evaluate_per_time_step(
        z_lf, support, default_probabilistic_obj_lst, time_vec, coords_mat
    )

    mp1.assert_called_once()
    mp1.assert_called_with(z_lf, time_vec, coords_mat)

    mp2.assert_called_once()
    mp2.assert_called_with(z_lf, 2, coords_mat)

    mp3.assert_called_once()
    mp3.assert_called_with(
        z_lf_array, support, num_coords, default_probabilistic_obj_lst, gradient_bool=False
    )

    np.testing.assert_array_equal(mean, default_mean)
    np.testing.assert_array_equal(variance, default_variance)


def test_prepare_z_lf_for_time_steps(default_bmfia_interface, mocker):
    """Test the preparation of the latent function for time steps."""
    # some inputs
    z_lf = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 2 x 2 x 2
    t_size = 2
    coords_mat = np.array([[0, 1], [0, 1]])  # one coordinates in 2d

    # call the method
    z_lf_out = BmfiaInterface._prepare_z_lf_for_time_steps(z_lf, t_size, coords_mat)

    # z_lf_out has 2 x 4 x 3
    z_lf_out_ref = np.array(
        [[[1, 0, 1], [3, 0, 1], [5, 0, 1], [7, 0, 1]], [[2, 0, 1], [4, 0, 1], [6, 0, 1], [8, 0, 1]]]
    )
    np.testing.assert_array_equal(z_lf_out, z_lf_out_ref)


def test_iterate_over_time_steps(default_bmfia_interface, mocker):
    """Test iterating over time steps."""
    # general inputs
    support = 'y'
    num_coords = 1
    gradient_bool = False

    # test wrong z_lf dimension
    z_lf = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        BmfiaInterface._iterate_over_time_steps(
            z_lf, support, num_coords, ["test", "test"], gradient_bool
        )

    # test iteration without gradient
    z_lf = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])

    mp1 = mocker.MagicMock()
    mp1.predict.return_value = {
        "mean": np.array([[1], [2]]),
        "variance": np.array([[3], [4]]),
        "grad_mean": np.array([[[5, 6], [6, 7]]]),
        "grad_var": np.array([[[7, 8], [8, 9]]]),
    }

    mp2 = mocker.MagicMock()
    mp2.predict.return_value = {
        "mean": np.array([[3], [4]]),
        "variance": np.array([[5], [6]]),
        "grad_mean": np.array([[[9, 10], [10, 11]]]),
        "grad_var": np.array([[[11, 12], [12, 13]]]),
    }
    mappings = [mp1, mp2]
    mean, variance, dummy_grad1, dummy_grad2 = BmfiaInterface._iterate_over_time_steps(
        z_lf, support, num_coords, mappings, gradient_bool
    )

    np.testing.assert_array_equal(mean, np.array([[1], [3], [2], [4]]))
    np.testing.assert_array_equal(variance, np.array([[3], [5], [4], [6]]))
    assert dummy_grad1 == []
    assert dummy_grad2 == []

    # test iteration with gradient
    gradient_bool = True
    mean, variance, grad_mean, grad_var = BmfiaInterface._iterate_over_time_steps(
        z_lf, support, num_coords, mappings, gradient_bool
    )
    np.testing.assert_array_equal(mean, np.array([[1], [3], [2], [4]]))
    np.testing.assert_array_equal(variance, np.array([[3], [5], [4], [6]]))
    np.testing.assert_array_equal(
        grad_mean, np.array([[[5], [6]], [[6], [7]], [[9], [10]], [[10], [11]]])
    )
    np.testing.assert_array_equal(
        grad_var, np.array([[[7], [8]], [[8], [9]], [[11], [12]], [[12], [13]]])
    )


def test_evaluate_and_gradient_per_coordinate(default_bmfia_interface, mocker):
    """Test the evaluation and gradient per coordinate."""
    # general inputs
    Z_LF = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    support = "y"

    # test evaluation per coordinate with correct 3d gradient
    map_1 = mocker.MagicMock()
    map_1.predict.return_value = {
        "mean": np.array([[1], [2]]),
        "variance": np.array([[3], [4]]),
        "grad_mean": np.array([[5, 6], [6, 7]]),
        "grad_var": np.array([[7, 8], [8, 9]]),
    }

    map_2 = mocker.MagicMock()
    map_2.predict.return_value = {
        "mean": np.array([[3], [4]]),
        "variance": np.array([[5], [6]]),
        "grad_mean": np.array([[9, 10], [10, 11]]),
        "grad_var": np.array([[11, 12], [12, 13]]),
    }

    probabilistic_mapping_obj_lst = [map_1, map_2]
    (mean, variance, grad_mean, grad_var,) = BmfiaInterface._evaluate_and_gradient_per_coordinate(
        Z_LF, support, probabilistic_mapping_obj_lst, None, None
    )

    assert map_1.predict.called_once()
    assert map_2.predict.called_once()

    np.testing.assert_array_equal(map_1.predict.call_args[0][0], Z_LF.T[0, :, :])
    assert map_1.predict.call_args[1]["support"] == support
    assert map_1.predict.call_args[1]["gradient_bool"] is True

    np.testing.assert_array_equal(map_2.predict.call_args[0][0], Z_LF.T[1, :, :])
    assert map_2.predict.call_args[1]["support"] == support
    assert map_2.predict.call_args[1]["gradient_bool"] is True

    np.testing.assert_array_equal(mean, np.array([[1, 3], [2, 4]]))
    np.testing.assert_array_equal(variance, np.array([[3, 5], [4, 6]]))

    # note that gradient is only selected w.r.t. the LF model
    np.testing.assert_array_equal(grad_mean, np.array([[5, 9], [6, 10]]))
    np.testing.assert_array_equal(grad_var, np.array([[7, 11], [8, 12]]))

    # test evaluation per coordinate with 2d gradient
    map_1 = mocker.MagicMock()
    map_1.predict.return_value = {
        "mean": np.array([[1], [2]]),
        "variance": np.array([[3], [4]]),
        "grad_mean": np.array([5, 6]),
        "grad_var": np.array([7, 8]),
    }

    map_2 = mocker.MagicMock()
    map_2.predict.return_value = {
        "mean": np.array([[3], [4]]),
        "variance": np.array([[5], [6]]),
        "grad_mean": np.array([9, 10]),
        "grad_var": np.array([11, 12]),
    }

    probabilistic_mapping_obj_lst = [map_1, map_2]
    (mean, variance, grad_mean, grad_var,) = BmfiaInterface._evaluate_and_gradient_per_coordinate(
        Z_LF, support, probabilistic_mapping_obj_lst, None, None
    )

    # note that gradient is only selected w.r.t. the LF model
    np.testing.assert_array_equal(grad_mean, np.array([[5, 9], [6, 10]]))
    np.testing.assert_array_equal(grad_var, np.array([[7, 11], [8, 12]]))


def test_evaluate_and_gradient_per_time_step(default_bmfia_interface, mocker):
    """Test the evaluation and gradient per time step."""
    # general inputs
    z_lf = np.array([[[1, 2], [3, 4]]])
    z_lf_array = np.array([[[1, 2, 3, 4]], [[5, 6, 7, 8]]])
    support = "y"
    default_mean = np.array([[7], [8]])
    default_variance = np.array([[11], [12]])
    default_grad_mean = np.array([[[1], [2]], [[3], [4]]])
    default_grad_variance = np.array([[[5], [6]], [[7], [8]]])
    default_bmfia_interface.time_vec = np.array([1, 2])  # two time steps
    default_bmfia_interface.coords_mat = np.array([1, 2])  # one coordinates in 2d
    num_coords = 1
    probabilistic_mapping_obj_lst = [mocker.MagicMock(), mocker.MagicMock()]
    time_vec = np.array([1, 2])
    coords_mat = np.array([1, 2])

    # mock check coordinate compliance
    mp1 = mocker.patch(
        "pqueens.interfaces.bmfia_interface.BmfiaInterface._check_coordinates_return_dimensions",
        return_value=(1, 2),
    )
    mp2 = mocker.patch(
        "pqueens.interfaces.bmfia_interface.BmfiaInterface._prepare_z_lf_for_time_steps",
        return_value=z_lf_array,
    )
    mp3 = mocker.patch(
        "pqueens.interfaces.bmfia_interface.BmfiaInterface._iterate_over_time_steps",
        return_value=(default_mean, default_variance, default_grad_mean, default_grad_variance),
    )

    (
        mean,
        variance,
        grad_mean,
        grad_variance,
    ) = BmfiaInterface._evaluate_and_gradient_per_time_step(
        z_lf, support, probabilistic_mapping_obj_lst, time_vec, coords_mat
    )

    mp1.assert_called_once()
    mp1.assert_called_with(z_lf, time_vec, coords_mat)

    mp2.assert_called_once()
    mp2.assert_called_with(z_lf, 2, coords_mat)

    mp3.assert_called_once()
    mp3.assert_called_with(
        z_lf_array, support, num_coords, probabilistic_mapping_obj_lst, gradient_bool=True
    )

    np.testing.assert_array_equal(mean, default_mean)
    np.testing.assert_array_equal(variance, default_variance)
    np.testing.assert_array_equal(grad_mean, default_grad_mean.swapaxes(1, 2))
    np.testing.assert_array_equal(grad_variance, default_grad_variance.swapaxes(1, 2))


def test_check_coordinates_return_dimensions():
    """Test the check coordinate compliance."""
    # test 1d time and coord inputs
    z_lf = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    time_vec = np.array([1, 2])  # two time steps
    coords_mat = np.array([1, 2])  # one coordinates in 2d
    num_coords, t_size = BmfiaInterface._check_coordinates_return_dimensions(
        z_lf, time_vec, coords_mat
    )

    assert num_coords == 1
    assert t_size == 2

    time_vec = np.array([[1, 2]])  # two time steps
    coords_mat = np.array([[1, 2]])  # one coordinates in 2d
    num_coords, t_size = BmfiaInterface._check_coordinates_return_dimensions(
        z_lf, time_vec, coords_mat
    )

    assert num_coords == 1
    assert t_size == 2

    # test wrong coordinate dimension --> column vector = 2 coords in 1 d
    coords_mat = np.array([[1, 2]]).T  # one coordinates in 2d
    with pytest.raises(ValueError):
        num_coords, t_size = BmfiaInterface._check_coordinates_return_dimensions(
            z_lf, time_vec, coords_mat
        )

    # test wrong number of time steps
    time_vec = np.array([[1, 2, 3]])  # three time steps
    coords_mat = np.array([[1, 2]])  # one coordinates in 2d
    with pytest.raises(ValueError):
        num_coords, t_size = BmfiaInterface._check_coordinates_return_dimensions(
            z_lf, time_vec, coords_mat
        )

    # test 2D Z_LF --> should be fine as well: first dim is gamma dim is just ignored here
    z_lf = np.array([[1, 2], [3, 4]])
    time_vec = np.array([1, 2])  # two time steps
    coords_mat = np.array([1, 2])  # one coordinates in 2d
    num_coords, t_size = BmfiaInterface._check_coordinates_return_dimensions(
        z_lf, time_vec, coords_mat
    )

    assert num_coords == 1
    assert t_size == 2


def test_update_mappings_per_coordinate(
    default_bmfia_interface,
    mocker,
    dummy_reg_obj,
):
    """Test the update mappings per coordinate."""
    z_lf_train = np.zeros((1, 2))
    y_hf_train = np.zeros((1, 2, 3))
    num_reg = y_hf_train.shape[2]
    default_bmfia_interface.probabilistic_mapping_obj_lst = []
    time_vec = None
    coords_mat = np.array([[0, 1], [0, 1]])

    # test wrong z_lf_train input dimensions
    with pytest.raises(IndexError):
        BmfiaInterface._update_mappings_per_coordinate(
            ["dummy", "dummy"], z_lf_train, y_hf_train, time_vec, coords_mat
        )

    # test correct z_lf_train input dimensions
    z_lf_train = np.zeros((1, 2, 3))
    mp1 = mocker.MagicMock()
    mp1.update_training_data.return_value = dummy_reg_obj
    probabilistic_mapping_obj_lst = [mp1] * num_reg

    (
        z_lf_array_out,
        y_hf_array_out,
        probabilistic_mapping_ob_lst,
    ) = BmfiaInterface._update_mappings_per_coordinate(
        probabilistic_mapping_obj_lst, z_lf_train, y_hf_train, time_vec, coords_mat
    )

    # --- asserts / tests
    assert probabilistic_mapping_obj_lst == probabilistic_mapping_ob_lst
    assert mp1.update_training_data.call_count == num_reg
    np.testing.assert_array_equal(z_lf_array_out, z_lf_train)
    np.testing.assert_array_equal(y_hf_array_out, y_hf_train)


def test_update_mappings_per_time_step(mocker):
    """Test the update mappings per time step."""
    z_lf_train = np.zeros((1, 2, 3))
    y_hf_train = np.zeros((2, 2, 2))
    z_lf_array_out = np.zeros((2, 1, 3))
    num_coords = 3
    t_size = 2
    time_vec = np.array([0, 1])
    coords_mat = np.array([[0, 1], [0, 1]])

    mp_1 = mocker.patch(
        "pqueens.interfaces.bmfia_interface.BmfiaInterface._check_coordinates_return_dimensions",
        return_value=(num_coords, t_size),
    )
    mp_2 = mocker.patch(
        "pqueens.interfaces.bmfia_interface.BmfiaInterface._prepare_z_lf_for_time_steps",
        return_value=z_lf_array_out,
    )

    mapping = mocker.MagicMock()
    mapping.update_training_data.return_value = dummy_reg_obj
    probabilistic_mapping_obj_lst = [mapping] * t_size

    (
        z_lf_array,
        y_hf_array,
        probabilistic_mapping_ob_lst,
    ) = BmfiaInterface._update_mappings_per_time_step(
        probabilistic_mapping_obj_lst, z_lf_train, y_hf_train, time_vec, coords_mat
    )

    # --- asserts / tests
    mp_1.assert_called_once()
    mp_1.assert_called_with(z_lf_train, time_vec, coords_mat)

    mp_2.assert_called_once()
    mp_2.assert_called_with(z_lf_train, t_size, coords_mat)

    assert mapping.update_training_data.call_count == 2
    np.testing.assert_array_equal(z_lf_array, z_lf_array_out)
    np.testing.assert_array_equal(y_hf_array, np.zeros((2, 4, 1)))
    assert probabilistic_mapping_obj_lst == probabilistic_mapping_ob_lst
