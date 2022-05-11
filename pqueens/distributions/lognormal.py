"""LogNormal Distribution."""
import numpy as np
import scipy.linalg
import scipy.stats

from pqueens.distributions import from_config_create_distribution
from pqueens.distributions.distributions import Distribution


class LogNormalDistribution(Distribution):
    """LogNormal distribution.

    Support in (0, +inf).

    Attributes:
        normal_mean (np.ndarray): mean of the underlying normal distribution
        normal_covariance (np.ndarray): covariance of the underlying normal distribution
        normal_distribution (NormalDistribution): underlying normal distribution
        logpdf_const (float): Constant for evaluation of log pdf
        precision (np.ndarray): Precision matrix of underlying normal distribution
    """

    def __init__(self, mean, covariance, normal_distribution):
        """Initialize lognormal distribution.

        Args:
            mean (np.ndarray): mean of the lognormal distribution
            covariance (np.ndarray): covariance of the lognormal distribution
            normal_distribution (np.ndarray): underlying normal distribution
        """
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
            distribution_options (dict): Dictionary with distribution description

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

        normal_covariance_diag = np.diag(normal_distribution.covariance)
        mean = np.exp(normal_distribution.mean + 0.5 * normal_covariance_diag)
        covariance = np.exp(
            normal_distribution.mean.reshape(-1, 1)
            + normal_distribution.mean.reshape(1, -1)
            + 0.5 * (normal_covariance_diag.reshape(-1, 1) + normal_covariance_diag.reshape(1, -1))
        ) * (np.exp(normal_distribution.covariance) - 1)

        return cls(mean=mean, covariance=covariance, normal_distribution=normal_distribution)

    def cdf(self, x):
        """Cumulative distribution function.

        Args:
            x (np.ndarray): Positions at which the cdf is evaluated

        Returns:
            cdf (np.ndarray): CDF at evaluated positions
        """
        return self.normal_distribution.cdf(np.log(x))

    def draw(self, num_draws=1):
        """Draw samples.

        Args:
            num_draws (int, optional): Number of draws

        Returns:
            samples (np.ndarray): Drawn samples from the distribution
        """
        return np.exp(self.normal_distribution.draw(num_draws=num_draws))

    def logpdf(self, x):
        """Log of the probability density function.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated

        Returns:
            logpdf (np.ndarray): log pdf at evaluated positions
        """
        log_x = np.log(x).reshape(-1, self.dimension)
        dist = log_x - self.normal_mean
        if dist.size == 1:  # TODO: Remove this if case, once autograd is removed
            logpdf = (self.logpdf_const - log_x[0, 0] - 0.5 * dist * self.precision * dist).reshape(
                -1
            )
        else:
            logpdf = (
                self.logpdf_const
                - np.sum(log_x, axis=1)
                - 0.5 * (np.dot(dist, self.precision) * dist).sum(axis=1)
            )
        return logpdf

    def pdf(self, x):
        """Probability density function.

        Args:
            x (np.ndarray): Positions at which the pdf is evaluated

        Returns:
            pdf (np.ndarray): pdf at evaluated positions
        """
        return np.exp(self.logpdf(x))

    def ppf(self, q):
        """Percent point function (inverse of cdf — quantiles).

        Args:
            q (np.ndarray): Quantiles at which the ppf is evaluated

        Returns:
            ppf (np.ndarray): Positions which correspond to given quantiles
        """
        self.check_1d()
        ppf = scipy.stats.lognorm.ppf(
            q, s=self.normal_covariance ** (1 / 2), scale=np.exp(self.normal_mean)
        ).reshape(-1)
        return ppf
