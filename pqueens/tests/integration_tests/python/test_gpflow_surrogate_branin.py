"""Integration test for GPflow based GP model."""

import pickle

import numpy as np
import pytest

from pqueens import run


def test_gpflow_surrogate_branin(
    inputdir, tmp_path, expected_mean, expected_variance, expected_posterior_samples
):
    """Test case for GPflow based GP model."""
    run(inputdir / 'gpflow_surrogate_branin.yml', tmp_path)

    result_file = tmp_path / 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["mean"], expected_mean, decimal=3
    )
    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["variance"], expected_variance, decimal=2
    )
    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["post_samples"], expected_posterior_samples, decimal=2
    )


@pytest.fixture()
def expected_mean():
    """Expected mean."""
    expected_mean = np.array(
        [
            [127.97233506],
            [39.73551321],
            [47.00641347],
            [28.88934819],
            [22.40199886],
            [150.69211917],
            [104.25630329],
            [92.22700928],
            [50.69060622],
            [22.18886383],
        ]
    )
    return expected_mean


@pytest.fixture()
def expected_variance():
    """Expected variance."""
    expected_variance = np.array(
        [
            [788.8004288],
            [1.8365012],
            [2.25043994],
            [4.24878946],
            [1.97026586],
            [174.50881662],
            [14.06623098],
            [8.34025715],
            [0.95922611],
            [0.33420735],
        ]
    )
    return expected_variance


@pytest.fixture()
def expected_posterior_samples():
    """Expected posterior samples."""
    expected_posterior_samples = np.array(
        [
            [1.890, 0.136, 2.294, -0.231, 0.461, 4.178, 0.695, 0.284, -0.265, 1.990],
            [0.375, -0.206, 1.942, 3.694, -0.037, 2.271, -0.222, 0.584, 3.734, 0.643],
            [0.444, 3.226, 0.662, 0.309, -0.227, 1.903, 2.412, -0.030, 2.387, -0.266],
        ]
    )
    return expected_posterior_samples.transpose()
