"""Test discrete particles distributions."""
import numpy as np
import pytest

from pqueens.distributions import from_config_create_distribution
from pqueens.distributions.particles import ParticleDiscreteDistribution


@pytest.fixture(name="reference_data", params=[1, 2])
def fixture_reference_data(request):
    """Data for the distribution."""
    reference_weights = [1, 2, 3, 4]
    reference_probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    if request.param == 1:
        return reference_weights, reference_probabilities, [[0], [1], [2], [3]], 1
    return reference_weights, reference_probabilities, [[1, 2], [3, 4], [2, 2], [0, 3]], 2


@pytest.fixture(name="distribution")
def fixture_distribution(reference_data):
    """Distribution fixture."""
    _, probabilities, sample_space, _ = reference_data
    return ParticleDiscreteDistribution(probabilities, sample_space)


def test_init_success(reference_data, distribution):
    """Test init method."""
    _, reference_probabilities, reference_sample_space, reference_dimension = reference_data

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
    np.testing.assert_allclose(reference_dimension, distribution.dimension)
    np.testing.assert_allclose(reference_mean, distribution.mean)
    np.testing.assert_allclose(reference_covariance, distribution.covariance)


def test_fcc(reference_data, distribution):
    """Test fcc function."""
    (
        reference_weights,
        reference_probabilities,
        reference_sample_space,
        reference_dimension,
    ) = reference_data

    distribution = from_config_create_distribution(
        {
            "type": "particles",
            "probabilities": reference_weights,
            "sample_space": reference_sample_space,
        }
    )
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
    np.testing.assert_allclose(reference_dimension, distribution.dimension)
    np.testing.assert_allclose(reference_mean, distribution.mean)
    np.testing.assert_allclose(reference_covariance, distribution.covariance)


def test_init_failure_mismatching_probabilities():
    """Test if mismatching number of probability leads to failure."""
    with pytest.raises(ValueError, match="The number of probabilities"):
        ParticleDiscreteDistribution([0.1, 1], [[1], [2], [3]])


def test_init_failure_negative_probabilities():
    """Test if negative values lead to errors."""
    with pytest.raises(ValueError, match="The parameter 'probabilities' has to be positive."):
        ParticleDiscreteDistribution([0.1, -1], [[1], [2]])


def test_init_failure_mismatching_dimension():
    """Test if mismatching dimensions of the event sa."""
    with pytest.raises(ValueError, match="Dimensions of the sample events do not match."):
        ParticleDiscreteDistribution([0.1, 1], [[1], [1, 2]])


def test_draw(mocker, reference_data, distribution):
    """Test draw."""
    _, _, reference_sample_space, _ = reference_data
    first_sample_event = 2
    second_sample_event = 3
    third_sample_event = 1
    mocker.patch(
        "pqueens.distributions.categorical.np.random.multinomial",
        return_value=np.array([first_sample_event, second_sample_event, third_sample_event]),
    )
    mocker.patch(
        "pqueens.distributions.categorical.np.random.shuffle",
    )
    reference_samples = [reference_sample_space[0]] * first_sample_event
    reference_samples.extend([reference_sample_space[1]] * second_sample_event)
    reference_samples.extend([reference_sample_space[2]] * third_sample_event)
    reference_samples = np.array(reference_samples, dtype=object).reshape(-1, 1)
    np.testing.assert_equal(reference_samples, distribution.draw(6))


def test_pdf(reference_data, distribution):
    """Test pdf."""
    _, reference_probabilities, reference_sample_space, _ = reference_data
    sample_location = np.array(reference_sample_space[::-1])
    np.testing.assert_allclose(reference_probabilities[::-1], distribution.pdf(sample_location))


def test_logpdf(reference_data, distribution):
    """Test logpdf."""
    _, reference_probabilities, reference_sample_space, _ = reference_data
    sample_location = np.array(reference_sample_space[::-1])
    np.testing.assert_allclose(
        np.log(reference_probabilities[::-1]), distribution.logpdf(sample_location)
    )


def test_cdf(reference_data, distribution):
    """Test cdf."""
    _, reference_probabilities, reference_sample_space, reference_dimension = reference_data
    sample_location = np.array(reference_sample_space[::-1])

    if reference_dimension == 1:
        np.testing.assert_allclose(
            np.cumsum(reference_probabilities)[::-1], distribution.cdf(sample_location)
        )
    else:
        with pytest.raises(ValueError, match="Method does not support multivariate distributions!"):
            distribution.cdf(sample_location)


def test_ppf(reference_data, distribution):
    """Test ppf."""
    _, reference_probabilities, reference_sample_space, reference_dimension = reference_data
    quantiles = np.array(np.cumsum(reference_probabilities)[::-1])

    if reference_dimension == 1:
        np.testing.assert_allclose(reference_sample_space[::-1], distribution.ppf(quantiles))
    else:
        with pytest.raises(ValueError, match="Method does not support multivariate distributions!"):
            distribution.ppf(quantiles)
