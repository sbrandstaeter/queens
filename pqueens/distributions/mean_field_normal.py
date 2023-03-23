"""Mean-field normal distribution."""
import numpy as np
import scipy.stats
from scipy.special import erf

from pqueens.distributions.distributions import Distribution


class MeanFieldNormalDistribution(Distribution):
    """Mean-field normal distribution.

    Attributes:
        standard_deviation (np.ndarray): standard deviation vector
    """

    def __init__(self, mean, variance, dimension):
        """Initialize normal distribution.

        Args:
            mean (np.ndarray): mean of the distribution
            variance (np.ndarray): variance of the distribution
            dimension (int): dimensionality of the distribution
        """
        mean = np.array(mean)
        variance = np.array(variance)
        mean = MeanFieldNormalDistribution.get_check_array_dimension_and_reshape(mean, dimension)
        covariance = MeanFieldNormalDistribution.get_check_array_dimension_and_reshape(
            variance, dimension
        )
        self.standard_deviation = np.sqrt(covariance)
        super().__init__(mean, covariance, dimension)

    def update_variance(self, variance):
        """Update the variance of the mean field distribution.

        Args:
            variance (np.array): New variance vector
        """
        covariance = MeanFieldNormalDistribution.get_check_array_dimension_and_reshape(
            variance, self.dimension
        )
        self.covariance = covariance
        self.standard_deviation = np.sqrt(covariance)

    def update_mean(self, mean):
        """Update the mean of the mean field distribution.

        Args:
            mean (np.array): New mean vector
        """
        mean = MeanFieldNormalDistribution.get_check_array_dimension_and_reshape(
            mean, self.dimension
        )
        self.mean = mean

    def cdf(self, x):
        """Cumulative distribution function.

        Args:
            x (np.ndarray): Positions at which the cdf is evaluated

        Returns:
            cdf (np.ndarray): cdf at evaluated positions
        """
        z = (x - self.mean) / self.standard_deviation
        cdf = 0.5 * (1 + erf(z / np.sqrt(2)))
        cdf = np.prod(cdf, axis=1).reshape(x.shape[0], -1)
        return cdf

    def draw(self, num_draws=1):
        """Draw samples.

        Args:
            num_draws (int, optional): Number of draws

        Returns:
            samples (np.ndarray): Drawn samples from the distribution
        """
        samples = np.random.randn(num_draws, self.dimension) * self.standard_deviation.reshape(
            1, -1
        ) + self.mean.reshape(1, -1)

        return samples

    def logpdf(self, x):
        """Log of the probability density function.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated

        Returns:
            logpdf (np.ndarray): log pdf at evaluated positions
        """
        dist = x - self.mean
        logpdf = (
            -0.5 * self.dimension * np.log(2 * np.pi)
            - 0.5 * np.sum(np.log(self.covariance))
            - 0.5 * np.sum(dist**2 / self.covariance, axis=1)
        ).flatten()

        return logpdf

    def grad_logpdf(self, x):
        """Gradient of the log pdf with respect to x.

        Args:
            x (np.ndarray): Positions at which the gradient of log pdf is evaluated

        Returns:
            grad_logpdf (np.ndarray): Gradient of the log pdf evaluated at positions
        """
        gradients_batch = -(x - self.mean) / self.covariance
        gradients_batch = gradients_batch.reshape(x.shape[0], -1)

        return gradients_batch

    def grad_logpdf_var(self, x):
        """Gradient of the log pdf with respect to the variance vector.

        Args:
            x (np.ndarray): Positions at which the gradient of log pdf is evaluated

        Returns:
            grad_logpdf_var (np.ndarray): Gradient of the log pdf w.r.t. the variance at given
                                        variance vector and position x
        """
        sample_batch = x.reshape(-1, self.dimension)

        part_1 = -0.5 * (1 / self.covariance)
        part_2 = 0.5 * ((sample_batch - self.mean) ** 2 / self.covariance**2)
        gradient_batch = part_1 + part_2

        grad_logpdf_var = gradient_batch.reshape(x.shape[0], -1)

        return grad_logpdf_var

    def pdf(self, x):
        """Probability density function.

        Args:
            x (np.ndarray): Positions at which the pdf is evaluated

        Returns:
            pdf (np.ndarray): pdf at evaluated positions
        """
        pdf = np.exp(self.logpdf(x))
        return pdf

    def ppf(self, q):
        """Percent point function (inverse of cdf — quantiles).

        Args:
            q (np.ndarray): Quantiles at which the ppf is evaluated
        """
        self.check_1d()
        ppf = scipy.stats.norm.ppf(q, loc=self.mean, scale=self.covariance ** (1 / 2)).reshape(-1)
        return ppf

    @staticmethod
    def get_check_array_dimension_and_reshape(input_array, dimension):
        """Check dimensions and potentially reshape array.

        Args:
            input_array (np.ndarray): Input array
            dimension (int): Dimension of the array

        Returns:
            input_array (np.ndarray): Input array with correct dimension
        """
        if not isinstance(input_array, np.ndarray):
            raise TypeError("Input must be a numpy array.")

        # allow one dimensional inputs that update the entire array
        if input_array.size == 1:
            input_array = np.tile(input_array, dimension)

        # raise error in case of dimension mismatch
        if input_array.size != dimension:
            raise ValueError(
                "Dimension of input vector and dimension attribute do not match."
                f"Provided dimension of input vector: {input_array.size}."
                f"Provided dimension was: {dimension}."
            )
        return input_array
