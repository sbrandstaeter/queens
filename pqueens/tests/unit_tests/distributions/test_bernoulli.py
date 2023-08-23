"""Test Bernoulli distributions."""
import numpy as np
import pytest

from pqueens.distributions.bernoulli import BernoulliDistribution


@pytest.fixture(name="reference_data")
def reference_data_fixture():
    """Data for the distribution."""
    success_probability = 0.3
    reference_probabilities = np.array([1 - success_probability, success_probability])
    reference_sample_space = np.array([[0], [1]])
    return success_probability, reference_sample_space, reference_probabilities


@pytest.fixture(name="distribution")
def distribution_fixture(reference_data):
    """Distribution fixture."""
    success_probability, _, _ = reference_data
    return BernoulliDistribution(success_probability)


def test_init_success(reference_data, distribution):
    """Test init method."""
    (
        reference_success_probability,
        reference_sample_space,
        reference_probabilities,
    ) = reference_data

    reference_mean = np.sum(
        [
            probability * np.array(value)
            for probability, value in zip(reference_probabilities, reference_sample_space)
        ],
        axis=0,
    )

    reference_covariance = np.sum(
        [
            probability * np.outer(value, value)
            for probability, value in zip(reference_probabilities, reference_sample_space)
        ],
        axis=0,
    ) - np.outer(reference_mean, reference_mean)

    np.testing.assert_allclose(reference_probabilities, distribution.probabilities)
    np.testing.assert_allclose(reference_sample_space, distribution.sample_space)
    np.testing.assert_allclose(1, distribution.dimension)
    np.testing.assert_allclose(reference_mean, distribution.mean)
    np.testing.assert_allclose(reference_covariance, distribution.covariance)
    np.testing.assert_allclose(reference_success_probability, distribution.success_probability)


@pytest.mark.parametrize("success_probability", [0, 1, -1])
def test_init_failure(success_probability):
    """Test if invalid options lead to errors."""
    with pytest.raises(
        ValueError,
        match="The success probability has to be 0<success_probability<1.",
    ):
        BernoulliDistribution(success_probability)
