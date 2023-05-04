"""Discrete uniform distribution."""
import numpy as np

from pqueens.distributions.particles import ParticleDiscreteDistribution


class UniformDiscreteDistribution(ParticleDiscreteDistribution):
    """Discrete uniform distribution."""

    def __init__(self, sample_space):
        """Initialize discrete uniform distribution.

        Args:
            sample_space (np.ndarray): Samples, i.e. possible outcomes of sampling the distribution
        """
        probabilities = np.ones(len(sample_space)) / len(sample_space)
        super().check_duplicates_in_sample_space(sample_space)
        super().__init__(probabilities, sample_space)
