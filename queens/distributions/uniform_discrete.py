"""Discrete uniform distribution."""

import numpy as np

from queens.distributions.particles import ParticleDiscreteDistribution
from queens.utils.logger_settings import log_init_args


class UniformDiscreteDistribution(ParticleDiscreteDistribution):
    """Discrete uniform distribution."""

    @log_init_args
    def __init__(self, sample_space):
        """Initialize discrete uniform distribution.

        Args:
            sample_space (np.ndarray): Samples, i.e. possible outcomes of sampling the distribution
        """
        probabilities = np.ones(len(sample_space)) / len(sample_space)
        super().check_duplicates_in_sample_space(sample_space)
        super().__init__(probabilities, sample_space)
