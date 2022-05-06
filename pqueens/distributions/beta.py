"""Beta Distribution."""
import scipy.linalg
import scipy.stats

from .distributions import Distribution


class BetaDistribution(Distribution):
    """Beta distribution.

    A generalized one dim beta distribution based on scipy stats. The generalized beta
    distribution has a lower bound and an upper bound.
    The parameters :math:`a` and :math:`b` determine the shape of the distribution within
    these bounds.

    Attributes:
        a (float): Shape parameter of the beta distribution, must be > 0
        b (float): Shape parameter of the beta distribution, must be > 0
        lower_bound (float): The lower bound of the beta distribution
        upper_bound (float): The upper bound of the beta distribution
    """

    def __init__(self, lower_bound, upper_bound, a, b):
        """Initialize Beta distribution."""
        # sanity checks:
        super().check_positivity({'a': a, 'b': b})
        super().check_bounds(lower_bound, upper_bound)

        self.a = a
        self.b = b
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.scale = upper_bound - lower_bound
        self.distribution = scipy.stats.beta(scale=self.scale, loc=lower_bound, a=a, b=b)
        super().__init__(
            mean=self.distribution.mean(), covariance=self.distribution.var(), dimension=1
        )

    @classmethod
    def from_config_create_distribution(cls, distribution_options):
        """Create beta distribution object from parameter dictionary.

        Args:
            distribution_options (dict):     Dictionary with distribution description

        Returns:
            distribution: BetaDistribution object
        """
        lower_bound = distribution_options['lower_bound']
        upper_bound = distribution_options['upper_bound']
        a = distribution_options['a']
        b = distribution_options['b']
        return cls(lower_bound=lower_bound, upper_bound=upper_bound, a=a, b=b)

    def cdf(self, x):
        """Cumulative distribution function."""
        return self.distribution.cdf(x)

    def draw(self, num_draws=1):
        """Draw samples."""
        return self.distribution.rvs(size=num_draws)

    def logpdf(self, x):
        """Log of the probability density function."""
        return self.distribution.logpdf(x)

    def pdf(self, x):
        """Probability density function."""
        return self.distribution.pdf(x)

    def ppf(self, q):
        """Percent point function (inverse of cdf — percentiles)."""
        return self.distribution.ppf(q)
