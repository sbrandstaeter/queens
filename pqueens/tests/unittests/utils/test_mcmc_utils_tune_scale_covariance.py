"""
Test-module for tune_cale_covariance of mcmc_utils module

@author: Sebastian Brandstaeter
"""

import pytest

from pqueens.utils.mcmc_utils import tune_scale_covariance

@pytest.fixture(
        scope='module',
        params=[
            (1e-4, 0.1),
            (1e-2, 0.5),
            (1e-1, 0.9),
            (4e-1, 1.0),
            (6e-1, 1.1),
            (8e-1, 2.0),
            (9.9e-1, 10.0)
        ]
)
def accept_rate_and_scale_covariance(request):
    """
    Return a set of valid acceptance rate and adjusted scale.

    given that the current scale is 1.0
    """
    return request.param


def test_tune_scale_covariance(accept_rate_and_scale_covariance):
    """ Test the tuning of scale covariance in MCMC methods. """

    accept_rate = accept_rate_and_scale_covariance[0]
    expected_scale = accept_rate_and_scale_covariance[1]
    current_scale = 1.0
    assert tune_scale_covariance(current_scale, accept_rate) == expected_scale
