"""Test-module for *tune_scale_covariance* of *mcmc_utils* module.

@author: Sebastian Brandstaeter
"""

import numpy as np
import pytest

from pqueens.utils.mcmc_utils import tune_scale_covariance


@pytest.fixture(
    name="accept_rate_and_scale_covariance",
    scope='module',
    params=[
        (1e-4, 0.1),
        (1e-2, 0.5),
        (1e-1, 0.9),
        (4e-1, 1.0),
        (6e-1, 1.1),
        (8e-1, 2.0),
        (9.9e-1, 10.0),
    ],
)
def fixture_accept_rate_and_scale_covariance(request):
    """Return a set of valid acceptance rate and adjusted scale.

    Given that the current scale is 1.0.
    """
    return request.param


def test_tune_scale_covariance(accept_rate_and_scale_covariance):
    """Test the tuning of scale covariance in MCMC methods."""
    accept_rate = accept_rate_and_scale_covariance[0]
    expected_scale = accept_rate_and_scale_covariance[1]
    current_scale = 1.0
    assert tune_scale_covariance(current_scale, accept_rate) == expected_scale


def test_tune_scale_covariance_multiple_chains():
    """TODO_doc: add a one-line explanation.

    Test the tuning of proposal covariance in MCMC methods with multiple
    chains.

    We assume here to have 7 parallel chains, that correspond to all
    possible tuning factors.
    """
    accept_rate = np.array([[1e-4, 1e-2, 1e-1, 4e-1, 6e-1, 8e-1, 9.9e-1]]).T
    expected_scale = np.array([[0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0]]).T

    current_scale = np.ones((7, 1))
    assert np.allclose(tune_scale_covariance(current_scale, accept_rate), expected_scale)
