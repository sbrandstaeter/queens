"""Unit tests for the simulation model."""
import numpy as np
import pytest
from mock import Mock

from pqueens.models.simulation_model import SimulationModel


# ------------------ some fixtures ------------------------------- #
@pytest.fixture()
def dummy_config():
    """A dummy config dictionary."""
    config = {"my_model": {"type": "simulation_model", "interface_name": "my_interface"}}
    return config


# ------------------ actual unit tests --------------------------- #
def test_init():
    """Test the init method of the simulation model."""
    interface = "my_interface"

    model_obj = SimulationModel(interface)
    assert model_obj.interface == interface


def test_fcc(dummy_config, mocker):
    """Test the fcc method."""
    model_name = "my_model"

    mocker.patch(
        "pqueens.models.simulation_model.from_config_create_interface", return_value="my_interface"
    )
    model_obj = SimulationModel.from_config_create_model(model_name, dummy_config)
    assert model_obj.interface == "my_interface"
    assert model_obj.__class__.__name__ == "SimulationModel"


def test_evaluate():
    """Test the evaluation method."""
    model_obj = SimulationModel(Mock())
    model_obj.interface.evaluate = lambda x: {"mean": x**2, "gradient": 2 * x}

    samples = np.array([[2.0]])
    response = model_obj.evaluate(samples)
    expected_response = {"mean": samples**2, "gradient": 2 * samples}
    assert response == expected_response
    assert model_obj.response == expected_response


def test_grad():
    """Test grad method."""
    model = SimulationModel('dummy_interface')
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
