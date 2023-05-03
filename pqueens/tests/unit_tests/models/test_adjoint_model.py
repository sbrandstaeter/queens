"""Unit tests for the adjoint model."""
from pathlib import Path

import numpy as np
import pytest
from mock import Mock

from pqueens.models import adjoint_model
from pqueens.models.adjoint_model import AdjointModel


# ------------------ some fixtures ------------------------------- #
@pytest.fixture()
def default_adjoint_model():
    """A default adjoint model."""
    model_obj = AdjointModel(
        model_name="my_model_name",
        global_settings={"experiment_name": "my_experiment"},
        interface=Mock(),
        gradient_interface=Mock(),
        adjoint_file="my_adjoint_file",
    )
    return model_obj


# ------------------ actual unit tests --------------------------- #
def test_init():
    """Test the init method of the adjoint model."""
    model_name = "my_model_name"
    global_settings = {"experiment_name": "my_experiment"}
    interface = "my_interface"
    gradient_interface = "my_gradient_interface"
    adjoint_file = "my_adjoint_file"

    # Test without grad handler
    model_obj = AdjointModel(
        model_name=model_name,
        global_settings=global_settings,
        interface=interface,
        gradient_interface=gradient_interface,
        adjoint_file=adjoint_file,
    )
    assert model_obj.name == model_name
    assert model_obj.experiment_name == global_settings["experiment_name"]
    assert model_obj.interface == interface
    assert model_obj.gradient_interface == gradient_interface
    assert model_obj.adjoint_file == adjoint_file


def test_evaluate(default_adjoint_model):
    """Test the evaluation method."""
    default_adjoint_model.interface.evaluate = lambda x: {"mean": x**2, "gradient": 2 * x}
    samples = np.array([[2.0]])
    response = default_adjoint_model.evaluate(samples)
    expected_response = {"mean": samples**2, "gradient": 2 * samples}
    assert response == expected_response
    assert default_adjoint_model.response == expected_response


def test_grad(default_adjoint_model):
    """Test grad method."""
    experiment_dir = Path('path_to_experiment_dir')
    adjoint_model.write_to_csv = Mock()
    default_adjoint_model.interface.job_ids = [1, 2, 3, 4, 5, 6]
    default_adjoint_model.gradient_interface.experiment_dir = experiment_dir
    default_adjoint_model.gradient_interface.evaluate = lambda x: {'mean': x**2}

    samples = np.random.random((2, 3))
    upstream_gradient = np.random.random((2, 4))
    gradient = np.random.random((2, 3, 4))
    default_adjoint_model.response = {"mean": None, "gradient": gradient}
    grad_out = default_adjoint_model.grad(samples, upstream_gradient=upstream_gradient)

    expected_grad = samples**2
    np.testing.assert_almost_equal(expected_grad, grad_out)

    assert adjoint_model.write_to_csv.call_count == 2
    assert (
        adjoint_model.write_to_csv.call_args_list[0].args[0]
        == experiment_dir / '5' / default_adjoint_model.adjoint_file
    )
    assert (
        adjoint_model.write_to_csv.call_args_list[1].args[0]
        == experiment_dir / '6' / default_adjoint_model.adjoint_file
    )
    np.testing.assert_equal(
        adjoint_model.write_to_csv.call_args_list[0].args[1], upstream_gradient[0].reshape(1, -1)
    )
    np.testing.assert_equal(
        adjoint_model.write_to_csv.call_args_list[1].args[1], upstream_gradient[1].reshape(1, -1)
    )
