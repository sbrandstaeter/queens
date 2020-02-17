"""
Test-module for Variables class

@author: Sebastian Brandstaeter
"""

import numpy as np
import pytest

from pqueens.variables.variables import Variables

# TODO this testing moduel is incomplete -> test all functionality of variables


@pytest.fixture(scope='module')
def data_vector():
    """ Possible data vector compatible with uncertain_parameters. """

    # total size is sum of size values of uncertain_parameters
    data_vector = np.zeros(4)

    data_vector[0] = 1.0
    data_vector[1] = 2.0
    data_vector[2] = 3.0
    data_vector[3] = 4.0

    return data_vector


@pytest.fixture(scope='module')
def uncertain_parameters(data_vector):
    """ Possible uncertain parameters dictionary. """

    uncertain_parameters = dict()
    uncertain_parameters['random_variables'] = dict()

    x1 = dict()
    x1['type'] = 'FLOAT'
    x1['size'] = 1
    x1['value'] = data_vector[0]
    x1['active'] = True
    # TODO test actual distributions
    x1['distribution'] = None

    x2 = dict()
    x2['type'] = 'FLOAT'
    x2['size'] = 2
    x2['value'] = data_vector[1:3]
    x2['active'] = True
    x2['distribution'] = None

    x3 = dict()
    x3['type'] = 'FLOAT'
    x3['size'] = 1
    x3['value'] = data_vector[3]
    x3['active'] = True
    x3['distribution'] = None

    uncertain_parameters['random_variables']['x1'] = x1
    uncertain_parameters['random_variables']['x2'] = x2
    uncertain_parameters['random_variables']['x3'] = x3

    return uncertain_parameters


@pytest.fixture(scope='module')
def active(data_vector):
    """ Possible active list. """

    active = [True] * len(data_vector)
    return active


@pytest.fixture()
def variable(uncertain_parameters, data_vector, active):
    """ An instance of an Variables class. """

    return Variables(uncertain_parameters, values=data_vector, active=active)


def test_from_data_vector_create(uncertain_parameters, data_vector):
    """
    Test from_uncertain_parameters_create
    """
    variable_instance = Variables.from_data_vector_create(uncertain_parameters, data_vector)
    data = variable_instance.get_active_variables_vector()

    np.testing.assert_allclose(data, data_vector)


def test_get_active_variables(variable, data_vector):
    """
    Test get_active_variables.
    """
    expected_active_variables_dict = {
        'x1': data_vector[0],
        'x2': data_vector[1:3],
        'x3': data_vector[3],
    }

    active_variables = variable.get_active_variables()

    for key, value in expected_active_variables_dict.items():
        np.testing.assert_allclose(active_variables[key], value)


def test_get_active_variables_vector(variable, data_vector):
    """
    Test get_active_variables_vector.
    """
    data = variable.get_active_variables_vector()

    np.testing.assert_allclose(data, data_vector)


def test_update_variables_from_vector(uncertain_parameters, data_vector, active):
    """ Test update_variables_from_vector method. """

    variable_i = Variables(uncertain_parameters, data_vector, active)
    new_vector = 2 * data_vector
    variable_i.update_variables_from_vector(new_vector)
    new_returned_vector = variable_i.get_active_variables_vector()

    np.testing.assert_allclose(new_vector, new_returned_vector)
