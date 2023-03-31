"""Test multinomial distribution."""
import numpy as np
import pytest

from pqueens.distributions import from_config_create_distribution
from pqueens.distributions.multinomial import MultinomialDistribution


@pytest.fixture(name="reference_data")
def fixture_reference_data():
    """Data for the distribution."""
    reference_n_trials = 10
    reference_probabilities = [0.1, 0.2, 0.3, 0.4]
    reference_dimension = 4
    return reference_n_trials, reference_probabilities, reference_dimension


@pytest.fixture(name="distribution")
def fixture_distribution(reference_data):
    """Distribution fixture."""
    n_trials, probabilities, _ = reference_data
    return MultinomialDistribution(n_trials, probabilities)


def test_init_success(reference_data, distribution):
    """Test init method."""
    reference_n_trials, reference_probabilities, reference_dimension = reference_data
    reference_mean = reference_n_trials * np.array(reference_probabilities)
    reference_covariance = reference_n_trials * (
        np.diag(reference_probabilities)
        - np.outer(reference_probabilities, reference_probabilities)
    )
    reference_sample_space = np.ones((reference_dimension, 1)) * reference_n_trials
    np.testing.assert_allclose(reference_probabilities, distribution.probabilities)
    np.testing.assert_allclose(reference_sample_space, distribution.sample_space)
    np.testing.assert_allclose(reference_dimension, distribution.dimension)
    np.testing.assert_allclose(reference_mean, distribution.mean)
    np.testing.assert_allclose(reference_covariance, distribution.covariance)


def test_fcc(reference_data, distribution):
    """Test fcc function."""
    reference_n_trials, reference_probabilities, reference_dimension = reference_data
    reference_mean = reference_n_trials * np.array(reference_probabilities)
    reference_covariance = reference_n_trials * (
        np.diag(reference_probabilities)
        - np.outer(reference_probabilities, reference_probabilities)
    )
    reference_sample_space = np.ones((reference_dimension, 1)) * reference_n_trials

    distribution = from_config_create_distribution(
        {
            "type": "multinomial",
            "probabilities": reference_probabilities,
            "n_trials": reference_n_trials,
        }
    )
    np.testing.assert_allclose(reference_probabilities, distribution.probabilities)
    np.testing.assert_allclose(reference_sample_space, distribution.sample_space)
    np.testing.assert_allclose(reference_dimension, distribution.dimension)
    np.testing.assert_allclose(reference_mean, distribution.mean)
    np.testing.assert_allclose(reference_covariance, distribution.covariance)


def test_draw(reference_data, distribution):
    """Test draw method."""
    reference_n_trials, _, reference_dimension = reference_data
    n_draws = 5
    samples = distribution.draw(n_draws)

    # assert if the shape is correct
    assert samples.shape == (n_draws, reference_dimension)

    # assert if the samples sum up to the number of trials
    np.testing.assert_equal(reference_n_trials, np.sum(samples, axis=1))


def test_pdf(distribution):
    """Test pdf."""
    locations = np.array([[1, 2, 3, 4], [10, 0, 0, 0]])
    reference_pdf = [0.03483648, 1e-10]
    np.testing.assert_allclose(reference_pdf, distribution.pdf(locations))


def test_logpdf(distribution):
    """Test logpdf."""
    locations = np.array([[1, 2, 3, 4], [10, 0, 0, 0]])
    reference_logpdf = np.log([0.03483648, 1e-10])
    np.testing.assert_allclose(reference_logpdf, distribution.logpdf(locations))


def test_cdf(distribution):
    """Test if cdf raises value error."""
    with pytest.raises(ValueError, match="Method does not support multivariate distributions!"):
        distribution.cdf(1)


def test_ppf(distribution):
    """Test if ppf raises value error."""
    with pytest.raises(ValueError, match="Method does not support multivariate distributions!"):
        distribution.ppf(1)
