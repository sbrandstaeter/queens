"""Free distribution."""

from pqueens.distributions.distributions import Distribution


class FreeDistribution(Distribution):
    """Free distribution class.

    This is not a proper distribution class. It is used for variables
    with no underlying distribution.
    """

    def __init__(self, dimension):
        """Initialize FreeDistribution object.

        Args:
            dimension (int): Dimensionality of the distribution
        """
        super().__init__(mean=None, covariance=None, dimension=dimension)

    @classmethod
    def from_config_create_distribution(cls, distribution_options):
        """Create FreeDistribution object from parameter dictionary.

        Args:
            distribution_options (dict): Dictionary with distribution description

        Returns:
            distribution: FreeDistribution object
        """
        dimension = distribution_options.get('lower_bound', 1)
        return cls(dimension=dimension)

    def cdf(self, _):
        """Cumulative distribution function."""
        raise NotImplementedError('cdf method is not implemented for FreeDistribution.')

    def draw(self, _):
        """Draw samples."""
        raise NotImplementedError('draw method is not implemented for FreeDistribution.')

    def logpdf(self, _):
        """Log of the probability density function."""
        raise NotImplementedError('logpdf method is not implemented for FreeDistribution.')

    def grad_logpdf(self, _):
        """Gradient of the log pdf with respect to *x*."""
        raise NotImplementedError('grad_logpdf method is not implemented for FreeDistribution.')

    def pdf(self, _):
        """Probability density function."""
        raise NotImplementedError('pdf method is not implemented for FreeDistribution.')

    def ppf(self, _):
        """Percent point function (inverse of cdf — quantiles)."""
        raise NotImplementedError('ppf method is not implemented for FreeDistribution.')
