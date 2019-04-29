"""
Test-module for abstract Model class

@author: Sebastian Brandstaeter
"""

import numpy as np
import pytest

from pqueens.models.model import Model


@pytest.fixture(scope='module')
def uncertain_parameters():
    """ Possible uncertain parameters dictionary. """

    uncertain_parameters = dict()
    uncertain_parameters['random_variables'] = dict()

    x1 = dict()
    x1['type'] = 'FLOAT'
    x1['size'] = 1

    x2 = dict()
    x2['type'] = 'FLOAT'
    x2['size'] = 2

    uncertain_parameters['random_variables']['x1'] = x1
    uncertain_parameters['random_variables']['x2'] = x2

    return uncertain_parameters


@pytest.fixture(scope='module')
def model_name():
    """ Model name as string. """

    return 'test_model'


@pytest.fixture()
def model(model_name, uncertain_parameters, mocker):
    """ An instance of an empty Model class. """
    # make abstract call Model instantiable
    mocker.patch.object(Model, '__abstractmethods__', new=set())

    return Model(model_name, uncertain_parameters)


@pytest.fixture(scope='module')
def data_vector():
    """ Possible data vector compatible with uncertain_parameters. """

    # total size is sum of size values of uncertain_parameters
    data_vector = np.zeros(3)

    data_vector[0] = 1.
    data_vector[1] = 2.
    data_vector[2] = 3.

    return data_vector


@pytest.fixture(scope='module')
def data_batch(data_vector):
    """ Possible data batch compatible with uncertain_parameters.

    A data batch is a collection of data vectors.
    """
    data_batch = np.stack([data_vector, data_vector])

    return data_batch


@pytest.fixture(scope='module')
def responses():
    """ Possible responses for vector valued model. """

    responses = dict()
    responses['mean'] = np.array([[1., 1.],
                                  [2., 2.]])

    return responses

def test_get_parameters(model, model_name, uncertain_parameters):
    """
    Test get_parameters
    """
    assert model.get_parameter() is uncertain_parameters


def test_update_model_from_sample(data_vector, model):
    """ Test if the model variables can be updated from ndarray. """

    model.update_model_from_sample(data_vector)

    assert len(model.variables) is 1
    np.testing.assert_allclose(model.variables[0].variables['x1']['value'], data_vector[0])
    np.testing.assert_allclose(model.variables[0].variables['x2']['value'], data_vector[1:])


def test_update_model_from_sample_batch(data_batch, model):
    """ Test if model variables can be updated from multiple ndarrays. """
    model.update_model_from_sample_batch(data_batch)

    assert len(model.variables) is 2

    for i, variables in enumerate(model.variables):
        np.testing.assert_allclose(variables.variables['x1']['value'], data_batch[i, 0])
        np.testing.assert_allclose(variables.variables['x2']['value'], data_batch[i, 1:])


def test_convert_array_to_model_variables(data_batch, model):
    """ Based on batch of data vectors return instance of variables. """

    variables = model.convert_array_to_model_variables(data_batch)

    assert len(variables) is 2

    for i, variable in enumerate(variables):
        np.testing.assert_allclose(variable.variables['x1']['value'], data_batch[i, 0])
        np.testing.assert_allclose(variable.variables['x2']['value'], data_batch[i, 1:])


def test_check_for_precalculated_response_of_sample_batch(data_batch, model, mocker):
    """ check if sample batch as already been evaluated. """

    # emulate the evaluation of the data_batch
    model.update_model_from_sample_batch(data_batch)

    # check that data_batch has indeed been already evaluated
    precalculated = model.check_for_precalculated_response_of_sample_batch(data_batch)

    assert precalculated is True


def test_check_for_precalculated_response_of_sample_batch_wrong_data(data_batch, model):
    """ Batch was NOT precalculated. """

    # emulate the evaluation of a different data_batch
    model.update_model_from_sample_batch(2 * data_batch)

    # the check should return False
    precalculated = model.check_for_precalculated_response_of_sample_batch(data_batch)

    assert precalculated is False

def test_check_for_precalculated_response_of_sample_batch_wrong_size(data_batch, model):
    """ Requested batch size does not match current variables size. """

    precalculated = model.check_for_precalculated_response_of_sample_batch(data_batch)

    assert precalculated is False

