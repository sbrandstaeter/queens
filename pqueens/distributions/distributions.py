"""Distributions."""
import abc


class Distribution:
    """Base class for probability distributions."""

    def __init__(self, mean, covariance, dimension):
        """Initialize proposal distribution."""
        self.mean = mean
        self.covariance = covariance
        self.dimension = dimension

    @abc.abstractmethod
    def cdf(self, x):
        """Evaluate the cumulative distribution function (cdf)."""
        pass

    @abc.abstractmethod
    def draw(self, num_draws=1):
        """Draw num_draws samples from distribution."""
        pass

    @abc.abstractmethod
    def logpdf(self, x):
        """Evaluate the natural logarithm of the pdf at sample."""
        pass

    @abc.abstractmethod
    def pdf(self, x):
        """Evaluate the probability density function (pdf) at sample."""
        pass

    @abc.abstractmethod
    def ppf(self, x):
        """Evaluate the ppf, i.e. the inverse of the cdf."""
        pass

    def check_1d(self):
        """Check if distribution is 1 dimensional."""
        if self.dimension != 1:
            raise RuntimeError("Method does not support multivariate distributions!")

    @staticmethod
    def check_positivity(parameters):
        """Check if parameters are positive."""
        for name, value in parameters.items():
            if value <= 0:
                raise ValueError(
                    f"The parameter {name} has to be positive. " f"You specified {name}={value}."
                )

    @staticmethod
    def check_bounds(lower_bound, upper_bound):
        """Check sanity of bounds."""
        if (upper_bound <= lower_bound).all():
            raise ValueError(
                f"Lower bound must be smaller than upper bound. "
                f"You specified lower_bound={lower_bound} and upper_bound={upper_bound}"
            )
