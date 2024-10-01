"""Unit tests for the simulation model."""

import numpy as np
import pytest
from mock import Mock

from queens.models.simulation_model import SimulationModel


# ------------------ actual unit tests --------------------------- #
def test_init():
    """Test the init method of the simulation model."""
    scheduler = Mock()
    driver = Mock()
    model_obj = SimulationModel(scheduler=scheduler, driver=driver)
    assert model_obj.scheduler == scheduler
    assert model_obj.driver == driver


def test_evaluate():
    """Test the evaluation method."""
    model_obj = SimulationModel(scheduler=Mock(), driver=Mock())
    model_obj.scheduler.evaluate = lambda x, driver: {"mean": x**2, "gradient": 2 * x}

    samples = np.array([[2.0]])
    response = model_obj.evaluate(samples)
    expected_response = {"mean": samples**2, "gradient": 2 * samples}
    assert response == expected_response
    assert model_obj.response == expected_response


def test_grad():
    """Test grad method."""
    model = SimulationModel(scheduler=Mock(), driver=Mock())
    np.random.seed(42)
    upstream_gradient = np.random.random((2, 4))
    gradient = np.random.random((2, 3, 4))
    model.response = {"mean": None, "gradient": gradient}
    grad_out = model.grad(None, upstream_gradient=upstream_gradient)
    expected_grad = np.sum(
        upstream_gradient[:, :, np.newaxis] * np.swapaxes(gradient, 1, 2), axis=1
    )
    np.testing.assert_almost_equal(expected_grad, grad_out)

    model.response = {"mean": None}
    with pytest.raises(ValueError):
        model.grad(None, upstream_gradient=upstream_gradient)
