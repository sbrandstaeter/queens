"""Utility methods used by the unit tests for distributions."""

import numpy as np


def covariance_discrete(probabilities, sample_space, mean):
    """Compute the covariance of a discrete distribution.

    Args:
        probabilities (np.ndarray): Probability of each event in the sample space
        sample_space (lst): Sample space of the discrete distribution holding the events
        mean (np.ndarray): Mean of the distribution

    Returns:
        np.ndarray: covariance
    """
    return np.sum(
        [
            probability * np.outer(value, value)
            for probability, value in zip(probabilities, sample_space)
        ],
        axis=0,
    ) - np.outer(mean, mean)
