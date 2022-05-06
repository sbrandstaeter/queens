"""Uniform distribution."""
import numpy as np
import scipy.linalg
import scipy.stats

from .distributions import Distribution


class UniformDistribution(Distribution):
    """Uniform distribution class.

    On the interval [a, b].
    """

    def __init__(self, lower_bound, upper_bound):
        """Initialize uniform distribution."""
        super().check_bounds(lower_bound, upper_bound)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.width = upper_bound - lower_bound

        mean = (self.lower_bound + self.upper_bound) / 2.0
        cov = self.width**2 / 12.0

        dim = 1
        super().__init__(mean=mean, covariance=cov, dimension=dim)

        self.pdf_const = 1.0 / self.width
        self.logpdf_const = np.log(self.pdf_const)

    @classmethod
    def from_config_create_distribution(cls, distribution_options):
        """Create beta distribution object from parameter dictionary.

        Args:
            distribution_options (dict):     Dictionary with distribution description

        Returns:
            distribution: UniformDistribution object
        """
        lower_bound = distribution_options['lower_bound']
        upper_bound = distribution_options['upper_bound']
        return cls(lower_bound=lower_bound, upper_bound=upper_bound)

    def cdf(self, x):
        """Cumulative distribution function."""
        return scipy.stats.uniform.cdf(x, loc=self.lower_bound, scale=self.width)

    def inverse_cdf(self, x):
        """Cumulative distribution function."""
        return x * self.width + self.lower_bound

    def draw(self, num_draws=1):
        """Draw samples."""
        return np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=num_draws)

    def logpdf(self, x):
        """Log of the probability density function."""
        if (x >= self.lower_bound) and (x <= self.upper_bound):
            return self.logpdf_const
        else:
            return -np.inf

    def pdf(self, x):
        """Probability density function."""
        if (x >= self.lower_bound) and (x <= self.upper_bound):
            return self.pdf_const
        else:
            return 0

    def ppf(self, q):
        """Percent point function (inverse of cdf — percentiles)."""
        return scipy.stats.uniform.ppf(q=q, loc=self.lower_bound, scale=self.width)
