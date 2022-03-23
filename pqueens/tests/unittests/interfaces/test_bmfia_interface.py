"""Unittests for the Bayesian multi-fidelity inverse analysis interface."""

import time

import numpy as np
import pytest

from pqueens.interfaces.bmfia_interface import BmfiaInterface


# ---- Fixtures and helper methods / classes ---------
@pytest.fixture
def default_bmfia_interface():
    """Fixture for a dummy bmfia interface."""
    config = {}
    approx_name = 'bmfia'
    num_processors_multi_processing = 2
    default_interface = BmfiaInterface(config, approx_name, num_processors_multi_processing)
    return default_interface


class DummyRegression:
    """A dummy regression class."""

    def __init__(self):
        """Init for dummy regression."""
        self.state = 0

    def predict(*_, **__):
        """A dummy predict method."""
        return {"mean": np.array([1, 2]), "variance": np.array([4, 5])}

    def train(*_, **__):
        """A dummpy training method."""
        time.sleep(0.01)

    def set_state(self, *_, **__):
        """A dummpy set_state method."""
        self.state = 1

    def get_state(self, *_, **__):
        """A dummpy get_state method."""
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
    """A dummy contect manager pool class."""

    def __init__(self, *_, **__):
        """Init context manager pool."""
        pass

    def __enter__(self):
        """Dummy enter method for context manager."""
        return self

    def __exit__(self, *_, **__):
        """Dummy exit method for context manager."""
        pass

    def map(self, *_, **__):
        """A dummy map method for the dummy pool."""
        return [1, 2, 3]

    def close(self):
        """A dummy close method."""
        pass


# ---- Actual unittests ------------------------------
@pytest.mark.unit_tests
def test__init__():
    """Test the instantiation of the interface object."""
    config = {'test': 'test'}
    approx_name = 'bmfia'
    num_processors_multi_processing = 3
    interface = BmfiaInterface(config, approx_name, num_processors_multi_processing)

    assert interface.config == config
    assert interface.approx_name == approx_name
    assert interface.probabilistic_mapping_obj_lst == []
    assert interface.num_processors_multi_processing == 3


@pytest.mark.unit_tests
def test_map(default_bmfia_interface, default_probabilistic_obj_lst):
    """Test the mapping for the multi-fidelity interface."""
    mean_in = np.array([[1, 1, 1], [2, 2, 2]])
    variance_in = np.array([[4, 4, 4], [5, 5, 5]])
    # Dims Z_LF: gamma_dim x num_samples x coord_dim
    #  --> here: 2 x 2 x 3
    Z_LF = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

    # --- Test empty probabilistic mapping list -----
    default_bmfia_interface.probabilistic_mapping_obj_lst = []
    with pytest.raises(RuntimeError):
        default_bmfia_interface.evaluate(Z_LF)

    # --- Test for differnt (wrong) dimensions of mapping list and z_lf
    default_bmfia_interface.probabilistic_mapping_obj_lst = default_probabilistic_obj_lst
    # Dims Z_LF: gamma_dim x num_samples x coord_dim
    #  --> here: 2 x 3
    Z_LF = np.array([[1, 2], [5, 6]])
    with pytest.raises(AssertionError):
        default_bmfia_interface.evaluate(Z_LF)

    # --- Test with correct list
    # Dims Z_LF: gamma_dim x num_samples x coord_dim
    #  --> here: 2 x 2 x 3
    Z_LF = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    default_bmfia_interface.probabilistic_mapping_obj_lst = default_probabilistic_obj_lst
    mean_out, variance_out = default_bmfia_interface.evaluate(Z_LF)

    # --- asserts / tests -------------
    np.testing.assert_array_equal(mean_in, mean_out)
    np.testing.assert_array_equal(variance_in, variance_out)


@pytest.mark.unit_tests
def test_build_approximation(default_bmfia_interface, mocker):
    """Test the set-up / build of the probabilsitic regression models."""
    Z_LF_train = np.zeros((2, 30))
    Y_HF_train = np.zeros((2, 25))
    dummy_lst = [1, 2, 3]

    # pylint: disable=line-too-long
    mock_instantiate = mocker.patch(
        'pqueens.interfaces.bmfia_interface.BmfiaInterface._instantiate_probabilistic_mappings'
    )
    mock_train_parallel = mocker.patch(
        'pqueens.interfaces.bmfia_interface.BmfiaInterface._train_probabilistic_mappings_in_parallel',
        return_value=dummy_lst,
    )
    mock_optimize_state = mocker.patch(
        'pqueens.interfaces.bmfia_interface.BmfiaInterface._set_optimized_state_of_probabilistic_mappings'
    )
    # pylint: disable=line-too-long

    # Test with wrong input dimensions --> AssertionError
    with pytest.raises(AssertionError):
        default_bmfia_interface.build_approximation(Z_LF_train, Y_HF_train)

    # Test with correct input dimensions
    Y_HF_train = np.zeros((2, 30))
    default_bmfia_interface.build_approximation(Z_LF_train, Y_HF_train)

    # -- Actual assert / tests ---
    assert mock_instantiate.call_once()
    assert mock_train_parallel.call_once()
    assert mock_optimize_state.call_once()

    np.testing.assert_array_equal(mock_instantiate.call_args[0][0], Z_LF_train)
    np.testing.assert_array_equal(mock_instantiate.call_args[0][1], Y_HF_train)
    np.testing.assert_array_equal(mock_train_parallel.call_args[0][0], Z_LF_train)
    np.testing.assert_array_equal(mock_optimize_state.call_args[0][0], dummy_lst)


@pytest.mark.unit_tests
def test_instantiate_probabilistic_mappings(
    default_bmfia_interface, mocker, dummy_reg_obj, default_probabilistic_obj_lst
):
    """Test the instantiation of the probabilistic mappings."""
    Z_LF_train = np.zeros((2, 3))
    Y_HF_train = np.zeros((2, 3))
    num_reg = Y_HF_train.shape[1]
    default_bmfia_interface.probabilistic_mapping_obj_lst = []

    mp_1 = mocker.patch(
        'pqueens.interfaces.bmfia_interface.from_config_create_regression_approximation',
        return_value=dummy_reg_obj,
    )
    default_bmfia_interface._instantiate_probabilistic_mappings(Z_LF_train, Y_HF_train)

    # --- asserts / tests
    assert default_probabilistic_obj_lst == default_bmfia_interface.probabilistic_mapping_obj_lst
    assert mp_1.call_count == num_reg
    assert mp_1.call_args[0][0] == default_bmfia_interface.config
    assert mp_1.call_args[0][1] == default_bmfia_interface.approx_name


@pytest.mark.unit_tests
def test_train_probabilistic_mappings_in_parallel(
    default_bmfia_interface, mocker, my_state_lst, default_probabilistic_obj_lst
):
    """Test the parallel training of the mappings."""
    Z_LF_train = np.zeros((2, 3))
    default_bmfia_interface.probabilistic_mapping_obj_lst = default_probabilistic_obj_lst
    mocker.patch('pqueens.interfaces.bmfia_interface.Pool', MyContextManagerPool)
    mo_2 = mocker.patch(__name__ + '.MyContextManagerPool.map', return_value=my_state_lst)

    # test with valid configuration
    default_bmfia_interface.num_processors_multi_processing = 3
    return_state_list = default_bmfia_interface._train_probabilistic_mappings_in_parallel(
        Z_LF_train
    )
    # --- asserts / tests ---
    assert my_state_lst == return_state_list
    assert mo_2.called_once()
    assert mo_2.call_args[0][0] == BmfiaInterface._optimize_hyper_params
    assert mo_2.call_args[0][1] == default_probabilistic_obj_lst

    # test with no specification for processors
    default_bmfia_interface.num_processors_multi_processing = None
    with pytest.raises(RuntimeError):
        default_bmfia_interface._train_probabilistic_mappings_in_parallel(Z_LF_train)

    # test with 0 as a specification for processors
    default_bmfia_interface.num_processors_multi_processing = 0
    with pytest.raises(RuntimeError):
        default_bmfia_interface._train_probabilistic_mappings_in_parallel(Z_LF_train)

    # test with float as a specification for processors
    default_bmfia_interface.num_processors_multi_processing = 1.3
    with pytest.raises(RuntimeError):
        default_bmfia_interface._train_probabilistic_mappings_in_parallel(Z_LF_train)

    # test with str as a specification for processors
    default_bmfia_interface.num_processors_multi_processing = 'blabla'
    with pytest.raises(RuntimeError):
        default_bmfia_interface._train_probabilistic_mappings_in_parallel(Z_LF_train)

    # test with too large number of processors
    default_bmfia_interface.num_processors_multi_processing = 99999999
    with pytest.raises(RuntimeError):
        default_bmfia_interface._train_probabilistic_mappings_in_parallel(Z_LF_train)


@pytest.mark.unit_tests
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


@pytest.mark.unit_tests
def test_optimize_hyper_params(mocker, dummy_reg_obj):
    """Test the training of a single mapping."""
    mo_1 = mocker.patch(__name__ + '.DummyRegression.train')
    state_dict = BmfiaInterface._optimize_hyper_params(dummy_reg_obj)

    # asserts / tests
    mo_1.assert_called_once()
    assert state_dict == {'test': 'test'}
