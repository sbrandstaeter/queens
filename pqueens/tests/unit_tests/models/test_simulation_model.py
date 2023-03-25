"""Unit tests for the simulation model."""
from collections import namedtuple

import numpy as np
import pytest

from pqueens.models.simulation_model import SimulationModel


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
