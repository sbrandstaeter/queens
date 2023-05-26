"""Unit tests for the simulation model."""
from collections import namedtuple

import numpy as np
import pytest

from pqueens.models.simulation_model import SimulationModel
from pqueens.utils.gradient_handler import ProvidedGradient


# ------------------ some fixtures ------------------------------- #
@pytest.fixture()
def dummy_config():
    """A dummy config dictionary."""
    config = {"my_model": {"interface_name": "my_interface"}}
    return config


# ------------------ actual unit tests --------------------------- #
def test_init():
    """Test the init method of the simulation model."""
    model_name = "my_model_name"
    interface = "my_interface"

    # Test without grad handler
    model_obj = SimulationModel(model_name, interface)
    assert model_obj.name == model_name
    assert model_obj.interface == interface
    assert model_obj.grad_handler is None
    assert model_obj.gradient_response is None

    # Test with grad handler
    grad_handler = "my_grad_handler"
    model_obj = SimulationModel(model_name, interface, grad_handler)
    assert model_obj.name == model_name
    assert model_obj.interface == interface
    assert model_obj.grad_handler == grad_handler
    assert model_obj.gradient_response is None


def test_fcc(dummy_config, mocker):
    """Test the fcc method."""
    model_name = "my_model"

    # test without gradient handler
    mocker.patch(
        "pqueens.models.simulation_model.from_config_create_interface", return_value="my_interface"
    )
    model_obj = SimulationModel.from_config_create_model(model_name, dummy_config)
    assert model_obj.name == model_name
    assert model_obj.interface == "my_interface"
    assert model_obj.grad_handler is None
    assert model_obj.gradient_response is None
    assert model_obj.__class__.__name__ == "SimulationModel"

    # test with gradient handler
    dummy_config[model_name]["gradient_handler_name"] = "my_grad_handler"
    mocker.patch(
        "pqueens.models.simulation_model.from_config_create_grad_handler",
        return_value="grad_handler",
    )
    model_obj = SimulationModel.from_config_create_model(model_name, dummy_config)
    assert model_obj.name == model_name
    assert model_obj.interface == "my_interface"
    assert model_obj.grad_handler == "grad_handler"
    assert model_obj.gradient_response is None
    assert model_obj.__class__.__name__ == "SimulationModel"


def test_evaluate():
    """Test the evaluation method."""
    interface_dummy = namedtuple("interface", ["resources", "evaluate"])
    model_name = "model_name"
    interface = interface_dummy("some_resource", lambda x: x**2)
    model_obj = SimulationModel(model_name, interface)

    samples = np.array([[2.0]])
    response = model_obj.evaluate(samples)
    assert response == 4.0
    assert model_obj.response == 4.0


def test_evaluate_and_gradient():
    """Test the evaluation and gradient method."""
    interface_dummy = namedtuple("interface", ["resources", "evaluate"])
    model_name = "model_name"

    def my_eval_fun(x):
        """Dummy eval function with gradient response."""
        y = x**2
        dy = 2 * x
        output = {"mean": y, "gradient": dy}
        return output

    interface = interface_dummy("some_resource", my_eval_fun)
    model_obj = SimulationModel(model_name, interface)
    samples = np.array([[1.0]])

    # test without grad_handler object
    with pytest.raises(AttributeError):
        response, gradient_response = model_obj.evaluate_and_gradient(samples)

    # test without grad_objective
    model_output_fun = ProvidedGradient._get_output_without_gradient_interface
    model_obj.grad_handler = ProvidedGradient("my_grad_handler", model_output_fun)
    response, gradient_response = model_obj.evaluate_and_gradient(samples)
    assert response == 1.0
    assert gradient_response == 2.0
    assert model_obj.response == 1.0
    assert model_obj.gradient_response == 2.0

    # test with grad_objective
    def gradient_objective(x, y):
        return 2 * y

    response, gradient_response = model_obj.evaluate_and_gradient(
        samples, upstream_gradient_fun=gradient_objective
    )
    assert response == 1.0
    assert gradient_response == 4.0
    assert model_obj.response == 1.0
    assert model_obj.gradient_response == 4.0
