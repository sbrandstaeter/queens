"""Test-module for distributions."""

import numpy as np
import pytest
import scipy.stats

from pqueens.distributions import from_config_create_distribution
from pqueens.distributions.normal import NormalDistribution
from pqueens.utils.valid_options_utils import InvalidOptionError


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
    with pytest.raises(InvalidOptionError, match=r'Requested distribution type not supported.*'):
        from_config_create_distribution(invalid_options)


@pytest.mark.unit_tests
def test_create_export_dict():
    """Test creation routine of distribution dict."""
    normal_options = {
        'distribution': 'normal',
        'mean': 0.0,
        'covariance': 1.0,
    }
    distribution = from_config_create_distribution(normal_options)
    exported_dict = distribution.export_dict()
    ref_dict = {
        'type': 'NormalDistribution',
        'mean': np.array([0.0]),
        'covariance': np.array([[1.0]]),
        'dimension': 1,
        'low_chol': np.array([[1.0]]),
        'precision': np.array([[1.0]]),
        'det_covariance': np.array(1.0),
        'logpdf_const': np.array([-0.9189385332046728]),
    }
    for (key, value), (key_ref, value_ref) in zip(exported_dict.items(), ref_dict.items()):
        assert key == key_ref
        np.testing.assert_equal(np.array(value), np.array(value_ref))
