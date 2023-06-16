"""Test-module for abstract Model class."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

import pqueens.parameters.parameters as parameters_module
from pqueens.models.model import Model


class TestModel(Model):
    def evaluate(self, samples):
        """Evaluate model with current set of samples."""

    def grad(self, samples, upstream_gradient):
        """Evaluate gradient of model with current set of samples."""


@pytest.fixture(scope='module')
def uncertain_parameters():
    """Possible uncertain parameters dictionary."""
    uncertain_parameters = {
        'x1': {'type': 'free', 'dimension': 1},
        'x2': {'type': 'free', 'dimension': 2},
    }

    parameters_module.from_config_create_parameters({"parameters": uncertain_parameters})
    return parameters_module.parameters


@pytest.fixture()
def model(uncertain_parameters):
    """An instance of an empty Model class."""
    return TestModel()


def test_init(model, uncertain_parameters):
    """Test init."""
    assert model.parameters is uncertain_parameters
    assert model.response is None
    assert model._evaluate_and_gradient_bool is False


def test_evaluate_and_gradient(model):
    """Test evaluate_and_gradient method."""
    assert model._evaluate_and_gradient_bool is False

    def model_eval(self, x):
        assert self._evaluate_and_gradient_bool is True
        return np.sum(x**2, axis=1, keepdims=True)

    model.grad = Mock(
        side_effect=lambda x, upstream_gradient: np.sum(
            upstream_gradient[:, :, np.newaxis] * 2 * x[:, np.newaxis, :], axis=1
        )
    )

    samples = np.random.random((3, 4))
    with patch.object(TestModel, "evaluate", new=model_eval):
        model_out, model_grad = model.evaluate_and_gradient(samples, upstream_gradient=None)
        assert model.grad.call_count == 1
        np.testing.assert_array_equal(model.grad.call_args.args[0], samples)
        np.testing.assert_array_equal(
            model.grad.call_args.kwargs['upstream_gradient'], np.ones((samples.shape[0], 1))
        )

        expected_model_out = np.sum(samples**2, axis=1, keepdims=True)
        expected_model_grad = 2 * samples
        np.testing.assert_array_equal(expected_model_out, model_out)
        np.testing.assert_array_equal(expected_model_grad, model_grad)

        # test with upstream_gradient
        upstream_ = np.random.random(samples.shape[0])
        model.evaluate_and_gradient(samples, upstream_gradient=upstream_)
        assert model.grad.call_count == 2
        np.testing.assert_array_equal(model.grad.call_args.args[0], samples)
        np.testing.assert_array_equal(
            model.grad.call_args.kwargs['upstream_gradient'], upstream_[:, np.newaxis]
        )

        assert model._evaluate_and_gradient_bool is False
