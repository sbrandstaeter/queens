"""Test discrete uniform distributions."""
import numpy as np
import pytest

from pqueens.distributions.uniform_discrete import UniformDiscreteDistribution


@pytest.fixture(name="reference_data", params=[1, 2])
def reference_data_fixture(request):
    """Data for the distribution."""
    reference_weights = [1, 1, 1, 1]
    reference_probabilities = np.array([0.25, 0.25, 0.25, 0.25])
    if request.param == 1:
        return reference_weights, reference_probabilities, [[0], [1], [2], [3]], 1

    return reference_weights, reference_probabilities, [[1, 2], [3, 4], [2, 2], [0, 3]], 2


@pytest.fixture(name="distribution")
def distribution_fixture(reference_data):
    """Distribution fixture."""
    _, _, sample_space, _ = reference_data
    return UniformDiscreteDistribution(sample_space)


@pytest.fixture(name="distribution_for_init", params=[0, 1])
def distribution_for_init_fixture(reference_data, distribution, request):
    """Distribution fixture."""
    if request.param == 0:
        return distribution

    (
        _,
        _,
        reference_sample_space,
        _,
    ) = reference_data

    return UniformDiscreteDistribution(sample_space=reference_sample_space)


def test_init_success(reference_data, distribution_for_init):
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

    np.testing.assert_allclose(reference_probabilities, distribution_for_init.probabilities)
    np.testing.assert_allclose(reference_sample_space, distribution_for_init.sample_space)
    np.testing.assert_allclose(reference_dimension, distribution_for_init.dimension)
    np.testing.assert_allclose(reference_mean, distribution_for_init.mean)
    np.testing.assert_allclose(reference_covariance, distribution_for_init.covariance)


@pytest.mark.parametrize("sample_space", [[1, 1, 2], [[1, 1], [2, 1], [1, 1]]])
def test_init_failure(sample_space):
    """Test if invalid options lead to errors."""
    with pytest.raises(ValueError, match="The sample space contains duplicate events"):
        UniformDiscreteDistribution(sample_space)
