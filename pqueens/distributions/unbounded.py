"""Unbounded distribution."""
import numpy as np

from pqueens.distributions.distributions import Distribution


class UnboundedDistribution(Distribution):
    """Unbounded distribution class."""

    def __init__(self, dimension):
        """Initialize uniform distribution.

        Args:
            dimension (int): Dimensionality of the distribution
        """
        super().__init__(mean=None, covariance=None, dimension=dimension)

    @classmethod
    def from_config_create_distribution(cls, distribution_options):
        """Create uniform distribution object from parameter dictionary.

        Args:
            distribution_options (dict): Dictionary with distribution description

        Returns:
            distribution: UnboundedDistribution object
        """
        dimension = distribution_options.get('dimension', 1)
        return cls(dimension=dimension)

    def cdf(self, x):
        """Cumulative distribution function.

        Args:
            x (np.ndarray): Positions at which the cdf is evaluated
        """
        raise NotImplementedError('cdf is not defined for UnboundedDistribution.')

    def draw(self, num_draws=1):
        """Draw samples.

        Args:
            num_draws (int, optional): Number of draws
        """
        raise NotImplementedError('Samples cannot be drawn from UnboundedDistribution.')

    def logpdf(self, x):
        """Log of the probability density function.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated
        """
        raise NotImplementedError('logpdf is not defined for UnboundedDistribution.')

    def grad_logpdf(self, x):
        """Gradient of the log pdf with respect to x.

        Args:
            x (np.ndarray): Positions at which the gradient of log pdf is evaluated
        """
        raise NotImplementedError('Gradient of logpdf is not defined for UnboundedDistribution.')

    def pdf(self, x):
        """Probability density function.

        Args:
            x (np.ndarray): Positions at which the pdf is evaluated
        """
        raise NotImplementedError('pdf is not defined for UnboundedDistribution.')

    def ppf(self, q):
        """Percent point function (inverse of cdf — quantiles).

        Args:
            q (np.ndarray): Quantiles at which the ppf is evaluated
        """
        raise NotImplementedError('ppf is not defined for UnboundedDistribution.')
