"""LogNormal Distribution."""
import numpy as np
import scipy.linalg
import scipy.stats

from pqueens.distributions import from_config_create_distribution

from .distributions import Distribution


class LogNormalDistribution(Distribution):
    """Lognormal distribution.

    normal_mean and normal_covariance are the parameters (mean and
    covariance) of the underlying normal distribution.
    """

    def __init__(self, mean, covariance, normal_distribution):
        """Initialize lognormal distribution."""
        super().__init__(mean=mean, covariance=covariance, dimension=normal_distribution.dimension)
        self.normal_mean = normal_distribution.mean
        self.normal_covariance = normal_distribution.covariance
        self.normal_distribution = normal_distribution
        self.logpdf_const = normal_distribution.logpdf_const
        self.precision = normal_distribution.precision

    @classmethod
    def from_config_create_distribution(cls, distribution_options):
        """Create beta distribution object from parameter dictionary.

        Args:
            distribution_options (dict):     Dictionary with distribution description

        Returns:
            distribution: BetaDistribution object
        """
        normal_mean = distribution_options['normal_mean']
        normal_covariance = distribution_options['normal_covariance']

        normal_distribution_dict = {
            'distribution': 'normal',
            'mean': normal_mean,
            'covariance': normal_covariance,
        }
        normal_distribution = from_config_create_distribution(normal_distribution_dict)

        normal_covariance_diag = np.diag(normal_covariance)
        mean = np.exp(normal_mean + 0.5 * normal_covariance_diag)
        covariance = np.exp(
            normal_mean.reshape(-1, 1)
            + normal_mean.reshape(1, -1)
            + 0.5 * (normal_covariance_diag.reshape(-1, 1) + normal_covariance_diag.reshape(1, -1))
        ) * (np.exp(normal_covariance) - 1)

        return cls(mean=mean, covariance=covariance, normal_distribution=normal_distribution)

    def cdf(self, x):
        """Cumulative distribution function."""
        return self.normal_distribution.cdf(np.log(x))

    def draw(self, num_draws=1):
        """Draw samples."""
        return np.exp(self.normal_distribution.draw(num_draws=num_draws))

    def logpdf(self, x):
        """Log of the probability density function."""
        log_x = np.log(x).reshape(-1, self.dimension)
        dist = log_x - self.normal_mean
        logpdf = (
            self.logpdf_const
            - np.sum(log_x, axis=1)
            - 0.5 * np.einsum('ij, jk, ik', dist, self.precision, dist)
        )
        return logpdf.reshape(-1)

    def pdf(self, x):
        """Probability density function."""
        return np.exp(self.logpdf(x))

    def ppf(self, q):
        """Percent point function (inverse of cdf — percentiles)."""
        if self.dimension == 1:
            ppf = scipy.stats.lognorm.ppf(
                q, s=self.normal_covariance ** (1 / 2), scale=np.exp(self.normal_mean)
            )
        else:
            raise RuntimeError(
                "ppf for multivariate gaussians is not supported.\n"
                "It is not uniquely defined, since cdf is not uniquely defined! "
            )
        return ppf
