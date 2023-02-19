"""Test-module for distributions."""

import numpy as np
import pytest

from pqueens.distributions import from_config_create_distribution
from pqueens.distributions.normal import NormalDistribution
from pqueens.utils.valid_options_utils import InvalidOptionError


def test_create_distribution_normal():
    """Test creation routine of distribution objects."""
    normal_options = {
        'type': 'normal',
        'mean': 0,
        'covariance': 1,
    }
    distribution = from_config_create_distribution(normal_options)
    assert isinstance(distribution, NormalDistribution)


def test_create_distribution_invalid():
    """Test creation routine of distribution objects."""
    invalid_options = {'type': 'UnsupportedType', 'lower_bound': 0.0, 'upper_bound': 1.0}
    with pytest.raises(InvalidOptionError):
        from_config_create_distribution(invalid_options)


def test_create_export_dict():
    """Test creation routine of distribution dict."""
    normal_options = {
        'type': 'normal',
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
        'logpdf_const': np.array([-0.9189385332046728]),
    }
    for (key, value), (key_ref, value_ref) in zip(exported_dict.items(), ref_dict.items()):
        assert key == key_ref
        if key == 'type':
            assert value == value_ref
        else:
            np.testing.assert_allclose(value, value_ref)
