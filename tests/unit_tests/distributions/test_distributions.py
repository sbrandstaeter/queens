"""Test-module for distributions."""

import numpy as np

from queens.distributions.normal import NormalDistribution


def test_create_export_dict():
    """Test creation routine of distribution dict."""
    distribution = NormalDistribution(mean=0.0, covariance=1.0)
    exported_dict = distribution.export_dict()
    ref_dict = {
        "type": "NormalDistribution",
        "mean": np.array([0.0]),
        "covariance": np.array([[1.0]]),
        "dimension": 1,
        "low_chol": np.array([[1.0]]),
        "precision": np.array([[1.0]]),
        "logpdf_const": np.array([-0.9189385332046728]),
    }
    for (key, value), (key_ref, value_ref) in zip(exported_dict.items(), ref_dict.items()):
        assert key == key_ref
        if key == "type":
            assert value == value_ref
        else:
            np.testing.assert_allclose(value, value_ref)
