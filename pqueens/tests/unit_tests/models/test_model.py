"""Test-module for abstract Model class.

@author: Sebastian Brandstaeter
"""

import numpy as np
import pytest

import pqueens.parameters.parameters as parameters_module
from pqueens.models.model import Model


@pytest.fixture(scope='module')
def uncertain_parameters():
    """Possible uncertain parameters dictionary."""
    uncertain_parameters = dict()

    x1 = dict()
    x1['type'] = 'random_variable'
    x1['dimension'] = 1

    x2 = dict()
    x2['type'] = 'random_variable'
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
    responses = dict()
    responses['mean'] = np.array([[1.0, 1.0], [2.0, 2.0]])

    return responses


def test_init(model, model_name, uncertain_parameters):
    """Test *get_parameters*."""
    assert model.parameters is uncertain_parameters
    assert model.name == model_name
    assert model.response is None
