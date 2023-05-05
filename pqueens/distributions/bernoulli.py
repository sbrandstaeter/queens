"""Bernoulli distribution."""
import numpy as np

from pqueens.distributions.particles import ParticleDiscreteDistribution


class BernoulliDistribution(ParticleDiscreteDistribution):
    """Bernoulli distribution."""

    def __init__(self, success_probability):
        """Initialize Bernoulli distribution.

        Args:
            success_probability (float): Probability of sampling 1
        """
        if success_probability <= 0 or success_probability >= 1:
            raise ValueError(
                "The success probability has to be 0<success_probability<1. You specified "
                f"success_probability={success_probability}"
            )
        sample_space = np.array([0, 1]).reshape(-1, 1)
        probabilities = np.array([1 - success_probability, success_probability])
        super().check_duplicates_in_sample_space(sample_space)
        super().__init__(probabilities, sample_space)

        self.success_probability = success_probability
