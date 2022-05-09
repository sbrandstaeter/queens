"""LogNormal Distribution."""
import numpy as np
import scipy.linalg
import scipy.stats

from .distributions import Distribution


class LogNormalDistribution(Distribution):
    """Lognormal distribution.

    mu and sigma are the parameters (mean and standard deviation) of the
    underlying normal distribution.
    """

    def __init__(self, mu, sigma):
        """Initialize lognormal distribution."""
        # sanity checks:
        super().check_positivity({'sigma': sigma})

        self.mu = mu
        self.sigma = sigma
        self.scipy_lognorm = scipy.stats.lognorm(scale=np.exp(self.mu), s=self.sigma)
        super().__init__(
            mean=self.scipy_lognorm.mean(), covariance=self.scipy_lognorm.var(), dimension=1
        )

        self.K1 = 1.0 / (self.sigma * np.sqrt(2 * np.pi))
        self.log_K1 = np.log(self.K1)
        self.K2 = 1.0 / (2.0 * self.sigma**2)

    @classmethod
    def from_config_create_distribution(cls, distribution_options):
        """Create beta distribution object from parameter dictionary.

        Args:
            distribution_options (dict):     Dictionary with distribution description

        Returns:
            distribution: BetaDistribution object
        """
        mu = distribution_options['mu']
        sigma = distribution_options['sigma']
        return cls(mu=mu, sigma=sigma)

    def cdf(self, x):
        """Cumulative distribution function."""
        return self.scipy_lognorm.cdf(x)

    def draw(self, num_draws=1):
        """Draw samples."""
        return np.random.lognormal(mean=self.mu, sigma=self.sigma, size=num_draws)

    def logpdf(self, x):
        """Log of the probability density function."""
        return -self.K2 * (np.log(x) - self.mu) ** 2 + self.log_K1 - np.log(x)

    def pdf(self, x):
        """Probability density function."""
        return self.K1 * np.exp(-self.K2 * (np.log(x) - self.mu) ** 2) / x

    def ppf(self, q):
        """Percent point function (inverse of cdf — percentiles)."""
        return scipy.stats.lognorm.ppf(q, s=self.sigma, scale=np.exp(self.mu))
