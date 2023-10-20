"""Unit tests for the adjoint model."""
from pathlib import Path

import numpy as np
import pytest
from mock import Mock

from queens.models import differentiable_simulation_model_adjoint
from queens.models.differentiable_simulation_model_adjoint import (
    DifferentiableSimulationModelAdjoint,
)


# ------------------ some fixtures ------------------------------- #
@pytest.fixture(name="default_adjoint_model")
def fixture_default_adjoint_model():
    """A default adjoint model."""
    model_obj = DifferentiableSimulationModelAdjoint(
        interface=Mock(),
        gradient_interface=Mock(),
        adjoint_file="my_adjoint_file",
    )
    return model_obj


# ------------------ actual unit tests --------------------------- #
def test_init():
    """Test the init method of the adjoint model."""
    interface = "my_interface"
    gradient_interface = "my_gradient_interface"
    adjoint_file = "my_adjoint_file"

    # Test without grad handler
    model_obj = DifferentiableSimulationModelAdjoint(
        interface=interface,
        gradient_interface=gradient_interface,
        adjoint_file=adjoint_file,
    )
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
    differentiable_simulation_model_adjoint.write_to_csv = Mock()
    default_adjoint_model.interface.latest_job_id = 6
    default_adjoint_model.gradient_interface.scheduler.experiment_dir = experiment_dir
    default_adjoint_model.gradient_interface.evaluate = lambda x: {'mean': x**2}

    np.random.seed(42)
    samples = np.random.random((2, 3))
    upstream_gradient = np.random.random((2, 4))
    gradient = np.random.random((2, 3, 4))
    default_adjoint_model.response = {"mean": None, "gradient": gradient}
    grad_out = default_adjoint_model.grad(samples, upstream_gradient=upstream_gradient)

    expected_grad = samples**2
    np.testing.assert_almost_equal(expected_grad, grad_out)

    assert differentiable_simulation_model_adjoint.write_to_csv.call_count == 2
    assert (
        differentiable_simulation_model_adjoint.write_to_csv.call_args_list[0].args[0]
        == experiment_dir / '5' / default_adjoint_model.adjoint_file
    )
    assert (
        differentiable_simulation_model_adjoint.write_to_csv.call_args_list[1].args[0]
        == experiment_dir / '6' / default_adjoint_model.adjoint_file
    )
    np.testing.assert_equal(
        differentiable_simulation_model_adjoint.write_to_csv.call_args_list[0].args[1],
        upstream_gradient[0].reshape(1, -1),
    )
    np.testing.assert_equal(
        differentiable_simulation_model_adjoint.write_to_csv.call_args_list[1].args[1],
        upstream_gradient[1].reshape(1, -1),
    )
