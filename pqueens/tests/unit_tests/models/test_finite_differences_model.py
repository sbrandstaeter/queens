"""Unit tests for the finite difference model."""

import numpy as np
import pytest
from mock import Mock

from pqueens.models.finite_difference_model import FiniteDifferenceModel
from pqueens.utils.valid_options_utils import InvalidOptionError


# ------------------ some fixtures ------------------------------- #
@pytest.fixture()
def default_fd_model():
    """A default finite difference model."""
    model_obj = FiniteDifferenceModel(
        model_name="my_model_name",
        interface=Mock(),
        finite_difference_method='2-point',
    )
    return model_obj


# ------------------ actual unit tests --------------------------- #
def test_init():
    """Test the init method of the finite difference model."""
    model_name = "my_model_name"
    interface = "my_interface"
    finite_difference_method = '3-point'
    step_size = 1e-6
    bounds = [-10, np.inf]

    model_obj = FiniteDifferenceModel(
        model_name=model_name,
        interface=interface,
        finite_difference_method=finite_difference_method,
        step_size=step_size,
        bounds=bounds,
    )
    assert model_obj.name == model_name
    assert model_obj.interface == interface
    assert model_obj.finite_difference_method == finite_difference_method
    assert model_obj.step_size == step_size
    np.testing.assert_equal(model_obj.bounds, np.array(bounds))

    with pytest.raises(InvalidOptionError):
        FiniteDifferenceModel(
            model_name=model_name,
            interface=interface,
            finite_difference_method='invalid_method',
            step_size=step_size,
        )


def test_evaluate(default_fd_model):
    """Test the evaluation method."""
    default_fd_model.interface.evaluate = lambda x: {"mean": np.sum(x**2, axis=1, keepdims=True)}
    samples = np.random.random((3, 2))

    expected_mean = np.sum(samples**2, axis=1, keepdims=True)
    expected_grad = 2 * samples[:, np.newaxis, :]

    response = default_fd_model.evaluate(samples)
    assert len(response) == 1
    np.testing.assert_array_equal(response['mean'], expected_mean)
    assert len(default_fd_model.response) == 1
    np.testing.assert_array_equal(default_fd_model.response['mean'], expected_mean)

    response = default_fd_model.evaluate(samples, gradient=False)
    assert len(response) == 1
    np.testing.assert_array_equal(response['mean'], expected_mean)
    assert len(default_fd_model.response) == 1
    np.testing.assert_array_equal(default_fd_model.response['mean'], expected_mean)

    response = default_fd_model.evaluate(samples, gradient=True)
    np.testing.assert_array_almost_equal(expected_mean, response['mean'], decimal=5)
    np.testing.assert_array_almost_equal(expected_grad, response['gradient'], decimal=5)

    default_fd_model.interface.evaluate = lambda x: {
        "mean": np.array([np.sum(x**2, axis=1), np.sum(2 * x**2, axis=1)]).T
    }
    samples = np.random.random((3, 4))

    expected_grad = np.swapaxes(np.array([2 * samples, 4 * samples]), 0, 1)
    expected_mean = np.array([np.sum(samples**2, axis=1), np.sum(2 * samples**2, axis=1)]).T
    response = default_fd_model.evaluate(samples, gradient=True)
    np.testing.assert_array_almost_equal(expected_mean, response['mean'], decimal=5)
    np.testing.assert_array_almost_equal(expected_grad, response['gradient'], decimal=4)


def test_grad(default_fd_model):
    """Test grad method."""
    samples = np.random.random((2, 4, 3))
    default_fd_model.response = {
        'mean': np.sum(samples**2, axis=2, keepdims=True),
        'gradient': 2 * samples,
    }
    upstream = np.random.random((2, 1))
    expected_grad = np.sum(
        upstream[:, :, np.newaxis] * default_fd_model.response['gradient'], axis=1
    )
    grad_out = default_fd_model.grad(samples, upstream)
    np.testing.assert_almost_equal(expected_grad, grad_out)
