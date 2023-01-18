"""Variable module."""


class Variable:
    """Variable class.

    This class is used for parameters with no underlying distribution.

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
