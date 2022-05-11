"""Test-module for distributions of mcmc_utils module."""

import numpy as np
import pytest
import scipy.stats

from pqueens.distributions import from_config_create_distribution
from pqueens.distributions.normal import NormalDistribution


@pytest.mark.unit_tests
def test_create_distribution_normal():
    """Test creation routine of distribution objects."""
    normal_options = {
        'distribution': 'normal',
        'mean': 0,
        'covariance': 1,
    }
    distribution = from_config_create_distribution(normal_options)
    assert isinstance(distribution, NormalDistribution)


@pytest.mark.unit_tests
def test_create_distribution_invalid():
    """Test creation routine of distribution objects."""
    invalid_options = {'distribution': 'UnsupportedType', 'lower_bound': 0.0, 'upper_bound': 1.0}
    with pytest.raises(ValueError, match=r'.*type not supported.*'):
        from_config_create_distribution(invalid_options)
