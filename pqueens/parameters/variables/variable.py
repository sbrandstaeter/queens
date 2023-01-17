"""Variable."""


class Variable:
    """Variable class.

    Attributes:
        dimension (int): Dimensionality of the variable
    """

    def __init__(self, dimension):
        """Initialize variable.

        Args:
            dimension (int): Dimensionality of the variable
        """
        self.dimension = dimension

    @classmethod
    def from_config_create_variable(cls, variable_options):
        """Create variable object from variable description.

        Args:
            variable_options (dict): Dictionary with variable description

        Returns:
            variable: Variable object
        """
        dimension = variable_options.get('dimension', 1)
        return cls(dimension=dimension)

    def cdf(self, x):
        """Cumulative distribution function.

        Args:
            x (np.ndarray): Positions at which the cdf is evaluated
        """
        raise NotImplementedError('cdf is not defined for Variable.')

    def draw(self, num_draws=1):
        """Draw samples.

        Args:
            num_draws (int, optional): Number of draws
        """
        raise NotImplementedError('Samples cannot be drawn from Variable.')

    def logpdf(self, x):
        """Log of the probability density function.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated
        """
        raise NotImplementedError('logpdf is not defined for Variable.')

    def grad_logpdf(self, x):
        """Gradient of the log pdf with respect to x.

        Args:
            x (np.ndarray): Positions at which the gradient of log pdf is evaluated
        """
        raise NotImplementedError('Gradient of logpdf is not defined for Variable.')

    def pdf(self, x):
        """Probability density function.

        Args:
            x (np.ndarray): Positions at which the pdf is evaluated
        """
        raise NotImplementedError('pdf is not defined for Variable.')

    def ppf(self, q):
        """Percent point function (inverse of cdf — quantiles).

        Args:
            q (np.ndarray): Quantiles at which the ppf is evaluated
        """
        raise NotImplementedError('ppf is not defined for Variable.')
