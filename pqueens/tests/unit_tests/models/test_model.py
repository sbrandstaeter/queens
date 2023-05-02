"""Test-module for abstract Model class.

@author: Sebastian Brandstaeter
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

import pqueens.parameters.parameters as parameters_module
from pqueens.models.model import Model


@pytest.fixture(scope='module')
def uncertain_parameters():
    """Possible uncertain parameters dictionary."""
    uncertain_parameters = {}

    x1 = {}
    x1['type'] = 'free'
    x1['dimension'] = 1

    x2 = {}
    x2['type'] = 'free'
    x2['dimension'] = 2

    uncertain_parameters['x1'] = x1
    uncertain_parameters['x2'] = x2

    parameters_module.from_config_create_parameters({"parameters": uncertain_parameters})
    return parameters_module.parameters


@pytest.fixture(scope='module')
def model_name():
    """Model name as string."""
    return 'test_model'


@pytest.fixture()
def model(model_name, uncertain_parameters, mocker):
    """An instance of an empty Model class."""
    # make abstract call Model instantiable
    mocker.patch.object(Model, '__abstractmethods__', new=set())

    return Model(model_name)


@pytest.fixture(scope='module')
def data_vector():
    """Possible data vector compatible with *uncertain_parameters*."""
    # total size is sum of size values of uncertain_parameters
    data_vector = np.zeros(3)

    data_vector[0] = 1.0
    data_vector[1] = 2.0
    data_vector[2] = 3.0

    return data_vector


@pytest.fixture(scope='module')
def data_batch(data_vector):
    """Possible data batch compatible with *uncertain_parameters*.

    A data batch is a collection of data vectors.
    """
    data_batch = np.stack([data_vector, data_vector])

    return data_batch


@pytest.fixture(scope='module')
def responses():
    """Possible responses for vector valued model."""
    responses = {}
    responses['mean'] = np.array([[1.0, 1.0], [2.0, 2.0]])

    return responses


def test_init(model, model_name, uncertain_parameters):
    """Test *get_parameters*."""
    assert model.parameters is uncertain_parameters
    assert model.name == model_name
    assert model.response is None
    assert model._evaluate_and_gradient_bool is False


def test_evaluate_and_gradient(model):
    """Test evaluate_and_gradient method."""
    assert model._evaluate_and_gradient_bool is False

    def model_eval(self, x):
        assert self._evaluate_and_gradient_bool is True
        return np.sum(x**2, axis=1, keepdims=True)

    model.grad = Mock(
        side_effect=lambda x, upstream: np.sum(
            upstream[:, :, np.newaxis] * 2 * x[:, np.newaxis, :], axis=1
        )
    )

    samples = np.random.random((3, 4))
    with patch.object(Model, "evaluate", new=model_eval):
        model_out, model_grad = model.evaluate_and_gradient(samples, upstream=None)
        assert model.grad.call_count == 1
        np.testing.assert_array_equal(model.grad.call_args.args[0], samples)
        np.testing.assert_array_equal(
            model.grad.call_args.kwargs['upstream'], np.ones((samples.shape[0], 1))
        )

        expected_model_out = np.sum(samples**2, axis=1, keepdims=True)
        expected_model_grad = 2 * samples
        np.testing.assert_array_equal(expected_model_out, model_out)
        np.testing.assert_array_equal(expected_model_grad, model_grad)

        # test with upstream
        upstream_ = np.random.random(samples.shape[0])
        model.evaluate_and_gradient(samples, upstream=upstream_)
        assert model.grad.call_count == 2
        np.testing.assert_array_equal(model.grad.call_args.args[0], samples)
        np.testing.assert_array_equal(
            model.grad.call_args.kwargs['upstream'], upstream_[:, np.newaxis]
        )

        assert model._evaluate_and_gradient_bool is False
