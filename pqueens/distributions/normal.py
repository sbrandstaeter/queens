"""Normal distribution."""
import numpy as np
import scipy.linalg
import scipy.stats

from pqueens.distributions.distributions import Distribution
from pqueens.utils import numpy_utils


class NormalDistribution(Distribution):
    """Normal distribution.

    Attributes:
        low_chol (np.ndarray): lower-triangular Cholesky factor of covariance matrix
        precision (np.ndarray): Precision matrix corresponding to covariance matrix
        det_covariance (float): Determinant of covariance matrix
        logpdf_const (float): Constant for evaluation of log pdf
    """

    def __init__(
        self, mean, covariance, dimension, low_chol, precision, det_covariance, logpdf_const
    ):
        """Initialize normal distribution.

        Args:
            mean (np.ndarray): mean of the distribution
            covariance (np.ndarray): covariance of the distribution
            dimension (int): dimensionality of the distribution
            low_chol (np.ndarray): lower-triangular Cholesky factor of covariance matrix
            precision (np.ndarray): Precision matrix corresponding to covariance matrix
            det_covariance (float): Determinant of covariance matrix
            logpdf_const (float): Constant for evaluation of log pdf
        """
        super().__init__(mean, covariance, dimension)
        self.low_chol = low_chol
        self.precision = precision
        self.det_covariance = det_covariance
        self.logpdf_const = logpdf_const

    @classmethod
    def from_config_create_distribution(cls, distribution_options):
        """Create beta distribution object from parameter dictionary.

        Args:
            distribution_options (dict): Dictionary with distribution description

        Returns:
            distribution: NormalDistribution object
        """
        mean = np.array(distribution_options['mean']).reshape(-1)
        covariance = numpy_utils.at_least_2d(np.array(distribution_options['covariance']))

        # sanity checks
        dimension = covariance.shape[0]
        if covariance.ndim != 2:
            raise ValueError(
                f"Provided covariance is not a matrix. "
                f"Provided covariance shape: {covariance.shape}"
            )
        if dimension != covariance.shape[1]:
            raise ValueError(
                "Provided covariance matrix is not quadratic. "
                f"Provided covariance shape: {covariance.shape}"
            )
        if not np.allclose(covariance.T, covariance):
            raise ValueError(
                "Provided covariance matrix is not symmetric. " f"Provided covariance: {covariance}"
            )
        if mean.shape[0] != dimension:
            raise ValueError(
                f"Dimension of mean vector and covariance matrix do not match. "
                f"Provided dimension of mean vector: {mean.shape[0]}. "
                f"Provided dimension of covariance matrix: {dimension}. "
            )

        low_chol = numpy_utils.safe_cholesky(covariance)

        # precision matrix Q and determinant of cov matrix
        chol_inv = np.linalg.inv(low_chol)
        precision = np.dot(chol_inv.T, chol_inv)

        # constant needed for pdf
        det_covariance = np.linalg.det(covariance)
        logpdf_const = -1 / 2 * np.log((2.0 * np.pi) ** dimension * det_covariance)

        return cls(
            mean=mean,
            covariance=covariance,
            dimension=dimension,
            low_chol=low_chol,
            precision=precision,
            det_covariance=det_covariance,
            logpdf_const=logpdf_const,
        )

    def cdf(self, x):
        """Cumulative distribution function.

        Args:
            x (np.ndarray): Positions at which the cdf is evaluated

        Returns:
            cdf (np.ndarray): CDF at evaluated positions
        """
        cdf = scipy.stats.multivariate_normal.cdf(
            x.reshape(-1, self.dimension), mean=self.mean, cov=self.covariance
        ).reshape(-1)
        return cdf

    def draw(self, num_draws=1):
        """Draw samples.

        Args:
            num_draws (int, optional): Number of draws

        Returns:
            samples (np.ndarray): Drawn samples from the distribution
        """
        uncorrelated_vector = np.random.randn(self.dimension, num_draws)
        samples = self.mean + np.dot(self.low_chol, uncorrelated_vector).T
        return samples

    def logpdf(self, x):
        """Log of the probability density function.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated

        Returns:
            logpdf (np.ndarray): log pdf at evaluated positions
        """
        dist = x.reshape(-1, self.dimension) - self.mean
        logpdf = self.logpdf_const - 0.5 * (np.dot(dist, self.precision) * dist).sum(axis=1)
        return logpdf

    def grad_logpdf(self, x):
        """Gradient of the log pdf.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated

        Returns:
            grad_logpdf (np.ndarray): Gradient of the log pdf evaluated at positions
        """
        x = x.reshape(-1, self.dimension)
        grad_logpdf = -0.5 * (
            np.dot(x, self.precision + self.precision.T)
            - np.dot(self.precision + self.precision.T, self.mean).reshape(1, self.dimension)
        )
        return grad_logpdf

    def ppf(self, q):
        """Percent point function (inverse of cdf — quantiles).

        Args:
            q (np.ndarray): Quantiles at which the ppf is evaluated

        Returns:
            ppf (np.ndarray): Positions which correspond to given quantiles
        """
        self.check_1d()
        ppf = scipy.stats.norm.ppf(q, loc=self.mean, scale=self.covariance).reshape(-1)
        return ppf
