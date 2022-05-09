"""Uniform distribution."""
import numpy as np
import scipy.linalg
import scipy.stats

from .distributions import Distribution


class UniformDistribution(Distribution):
    """Uniform distribution class.

    On the interval [a, b].
    """

    def __init__(
        self, lower_bound, upper_bound, width, pdf_const, logpdf_const, mean, covariance, dimension
    ):
        """Initialize uniform distribution."""
        super().__init__(mean=mean, covariance=covariance, dimension=dimension)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.width = width
        self.pdf_const = pdf_const
        self.logpdf_const = logpdf_const

    @classmethod
    def from_config_create_distribution(cls, distribution_options):
        """Create beta distribution object from parameter dictionary.

        Args:
            distribution_options (dict):     Dictionary with distribution description

        Returns:
            distribution: UniformDistribution object
        """
        lower_bound = np.array(distribution_options['lower_bound']).reshape(-1)
        upper_bound = np.array(distribution_options['upper_bound']).reshape(-1)
        super().check_bounds(lower_bound, upper_bound)
        width = upper_bound - lower_bound

        mean = (lower_bound + upper_bound) / 2.0
        covariance = np.diag(width**2 / 12.0)
        dimension = mean.size

        pdf_const = 1.0 / width
        logpdf_const = np.log(pdf_const)

        return cls(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            width=width,
            pdf_const=pdf_const,
            logpdf_const=logpdf_const,
            mean=mean,
            covariance=covariance,
            dimension=dimension,
        )

    def cdf(self, x):
        """Cumulative distribution function."""
        cdf = np.prod(
            np.clip(
                (x.reshape(-1, self.dimension) - self.lower_bound) / self.width,
                a_min=np.zeros(self.dimension),
                a_max=np.ones(self.dimension),
            )
        )
        return cdf

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
        self.check_1d()
        return scipy.stats.uniform.ppf(q=q, loc=self.lower_bound, scale=self.width)
