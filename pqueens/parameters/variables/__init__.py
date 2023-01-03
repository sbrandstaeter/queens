"""Random Variables."""

from pqueens.distributions import from_config_create_distribution
from pqueens.parameters.variables.random_variables import RandomVariable


def from_config_create_random_variable(rv_dict):
    """Create random variables from problem description.

    Args:
        rv_dict (dict):       Dictionary with QUEENS problem description

    Returns:
        rv: Random variables object
    """

    if rv_dict.get("distribution"):
        distribution = from_config_create_distribution(rv_dict)
        dimension = distribution.dimension
    else:
        distribution = None
        dimension = rv_dict['dimension']
    lower_bound = rv_dict.get('lower_bound', None)
    upper_bound = rv_dict.get('upper_bound', None)
    return RandomVariable(distribution, dimension, lower_bound, upper_bound)
