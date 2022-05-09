"""Normal distribution."""
import numpy as np
import scipy.linalg
import scipy.stats

from pqueens.utils import numpy_utils

from .distributions import Distribution


class NormalDistribution(Distribution):
    """Normal distribution."""

    def __init__(self, mean, covariance, dimension, low_chol, precision, det_cov, logpdf_const):
        """Initialize normal distribution."""
        super().__init__(mean, covariance, dimension)
        self.low_chol = low_chol
        self.precision = precision
        self.det_cov = det_cov
        self.logpdf_const = logpdf_const

    @classmethod
    def from_config_create_distribution(cls, distribution_options):
        """Create beta distribution object from parameter dictionary.

        Args:
            distribution_options (dict):     Dictionary with distribution description

        Returns:
            distribution: NormalDistribution object
        """
        mean = np.array(distribution_options['mean']).reshape(-1)
        covariance = numpy_utils.at_least_2d(np.array(distribution_options['covariance']))

        # sanity checks
        dimension = covariance.shape[0]
        if covariance.ndim != 2:
            raise ValueError("Provided covariance is not a matrix.")
        if dimension != covariance.shape[1]:
            raise ValueError("Provided covariance matrix is not quadratic.")
        if not np.allclose(covariance.T, covariance):
            raise ValueError("Provided covariance matrix is not symmetric.")
        if mean.shape[0] != dimension:
            raise ValueError("Dimension of mean vector and covariance vector do not match.")

        low_chol = numpy_utils.safe_cholesky(covariance)

        # precision matrix Q and determinant of cov matrix
        chol_inv = np.linalg.inv(low_chol)
        precision = np.dot(chol_inv.T, chol_inv)

        # constant needed for pdf
        det_cov = np.linalg.det(covariance)
        logpdf_const = -1 / 2 * np.log((2.0 * np.pi) ** dimension * det_cov)

        return cls(
            mean=mean,
            covariance=covariance,
            dimension=dimension,
            low_chol=low_chol,
            precision=precision,
            det_cov=det_cov,
            logpdf_const=logpdf_const,
        )

    def cdf(self, x):
        """Cumulative distribution function."""
        cdf = scipy.stats.multivariate_normal.cdf(x, mean=self.mean, cov=self.covariance)
        return cdf

    def draw(self, num_draws=1):
        """Draw samples."""
        uncorrelated_vector = np.random.randn(self.dimension, num_draws)
        samples = self.mean + np.dot(self.low_chol, uncorrelated_vector).T
        return samples

    def logpdf(self, x):
        """Log of the probability density function."""
        y = x.reshape(-1, self.dimension) - self.mean
        if y.shape[0] == 1:  # This is only needed because np.einsum can not handle autograd objects
            logpdf = self.logpdf_const - 0.5 * np.dot(np.dot(y, self.precision), y.T).reshape(-1)
        else:
            logpdf = self.logpdf_const - 0.5 * np.einsum(
                'ij, jk, ik -> i', y, self.precision, y
            ).reshape(-1)
        return logpdf

    def ppf(self, q):
        """Percent point function (inverse of cdf — percentiles)."""
        self.check_1d()
        ppf = scipy.stats.norm.ppf(q, loc=self.mean.squeeze(), scale=self.covariance.squeeze())
        return ppf

    def pdf(self, x):
        """Probability density function."""
        pdf = np.exp(self.logpdf(x))
        return pdf
