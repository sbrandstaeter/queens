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
    default_interface = BmfiaInterface(config, approx_name)
    return default_interface


class DummyRegression:
    """A dummy regression class."""

    def predict(*_, **__):
        """A dummy predict method."""
        return {"mean": np.array([1, 2]), "variance": np.array([4, 5])}

    def train(*_, **__):
        """A dummpy training method."""
        time.sleep(0.01)


@pytest.fixture
def dummy_reg_obj():
    """Fixture for a dummy regression object."""
    obj = DummyRegression()
    return obj


@pytest.fixture
def default_probabilistic_obj_lst(dummy_reg_obj):
    """Fixture for probabilistic mapping objects."""
    dummy_lst = [dummy_reg_obj, dummy_reg_obj, dummy_reg_obj]
    return dummy_lst


# ---- Actual unittests ------------------------------
@pytest.mark.unit_tests
def test__init__():
    """Test the instantiation of the interface object."""
    config = {'test': 'test'}
    approx_name = 'bmfia'
    interface = BmfiaInterface(config, approx_name)

    assert interface.config == config
    assert interface.approx_name == approx_name
    assert interface.probabilistic_mapping_obj_lst == []


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
        default_bmfia_interface.map(Z_LF, support='y', full_cov=False)

    # --- Test for differnt (wrong) dimensions of mapping list and z_lf
    default_bmfia_interface.probabilistic_mapping_obj_lst = default_probabilistic_obj_lst
    # Dims Z_LF: gamma_dim x num_samples x coord_dim
    #  --> here: 2 x 3
    Z_LF = np.array([[1, 2], [5, 6]])
    with pytest.raises(AssertionError):
        default_bmfia_interface.map(Z_LF, support='y', full_cov=False)

    # --- Test with correct list
    # Dims Z_LF: gamma_dim x num_samples x coord_dim
    #  --> here: 2 x 2 x 3
    Z_LF = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    default_bmfia_interface.probabilistic_mapping_obj_lst = default_probabilistic_obj_lst
    mean_out, variance_out = default_bmfia_interface.map(Z_LF, support='y', full_cov=False)

    # --- asserts / tests -------------
    np.testing.assert_array_equal(mean_in, mean_out)
    np.testing.assert_array_equal(variance_in, variance_out)


@pytest.mark.unit_tests
def test_build_approximation(default_bmfia_interface, mocker, dummy_reg_obj):
    """Test the set-up / build of the probabilsitic regression models."""
    Z_LF_train = np.zeros((2, 30))
    Y_HF_train = np.zeros((2, 30))
    num_reg_obj = Z_LF_train.T.shape[0]

    # pylint: disable=line-too-long
    mo_1 = mocker.patch(
        'pqueens.regression_approximations.regression_approximation.RegressionApproximation.from_config_create',
        return_value=dummy_reg_obj,
    )
    # pylint: disable=line-too-long

    t_start = time.time()
    default_bmfia_interface.build_approximation(Z_LF_train, Y_HF_train)
    t_end = time.time()
    t_diff = t_end - t_start

    # -- Actual assert / tests ---
    assert mo_1.call_count == num_reg_obj
    assert t_diff < 0.01 * Z_LF_train.shape[1]
    # TODO implement and test parallelity
    # TODO test assert statement of non-compliant dimensions
