"""Distributions."""
import abc

import numpy as np

from pqueens.utils.print_utils import get_str_table


class Distribution:
    """Base class for probability distributions.

    Attributes:
        mean (np.ndarray): Mean of the distribution.
        covariance (np.ndarray): Covariance of the distribution.
        dimension (int): Dimensionality of the distribution.
    """

    def __init__(self, mean, covariance, dimension):
        """Initialize distribution.

        Args:
            mean (np.ndarray): mean of the distribution
            covariance (np.ndarray): covariance of the distribution
            dimension (int): dimensionality of the distribution
        """
        self.mean = mean
        self.covariance = covariance
        self.dimension = dimension

    @classmethod
    def from_config_create_distribution(cls, distribution_options):
        """Create distribution object from parameter dictionary.

        Args:
            distribution_options (dict): Dictionary with distribution description

        Returns:
            distribution: Distribution object
        """
        distribution_options_copy = distribution_options.copy()
        distribution_options_copy.pop("type", None)
        return cls(**distribution_options_copy)

    @abc.abstractmethod
    def cdf(self, x):
        """Cumulative distribution function.

        Args:
            x (np.ndarray): Positions at which the cdf is evaluated
        """

    @abc.abstractmethod
    def draw(self, num_draws=1):
        """Draw samples.

        Args:
            num_draws (int, optional): Number of draws
        """

    @abc.abstractmethod
    def logpdf(self, x):
        """Log of the probability density function.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated
        """

    @abc.abstractmethod
    def grad_logpdf(self, x):
        """Gradient of the log pdf with respect to *x*.

        Args:
            x (np.ndarray): Positions at which the gradient of log pdf is evaluated
        """

    @abc.abstractmethod
    def pdf(self, x):
        """Probability density function.

        Args:
            x (np.ndarray): Positions at which the pdf is evaluated
        """

    @abc.abstractmethod
    def ppf(self, q):
        """Percent point function (inverse of cdf — quantiles).

        Args:
            q (np.ndarray): Quantiles at which the ppf is evaluated
        """

    def check_1d(self):
        """Check if distribution is one-dimensional."""
        if self.dimension != 1:
            raise ValueError("Method does not support multivariate distributions!")

    @staticmethod
    def check_positivity(**parameters):
        """Check if parameters are positive.

        Args:
            parameters (dict): Checked parameters
        """
        for name, value in parameters.items():
            if (np.array(value) <= 0).any():
                raise ValueError(
                    f"The parameter \'{name}\' has to be positive. "
                    f"You specified {name}={value}."
                )

    @staticmethod
    def check_bounds(lower_bound, upper_bound):
        """Check sanity of bounds.

        Args:
            lower_bound (np.ndarray): Lower bound(s) of distribution
            upper_bound (np.ndarray): Upper bound(s) of distribution
        """
        if (upper_bound <= lower_bound).any():
            raise ValueError(
                f"Lower bound must be smaller than upper bound. "
                f"You specified lower_bound={lower_bound} and upper_bound={upper_bound}"
            )

    def export_dict(self):
        """Create a dict of the distribution.

        Returns:
            export_dict (dict): Dict containing distribution information
        """
        export_dict = vars(self)
        export_dict = {'type': self.__class__.__name__, **export_dict}
        return export_dict

    def __str__(self):
        """Get string for the given distribution.

        Returns:
            str: Table with distribution information
        """
        return get_str_table(type(self).__name__, self.export_dict())
