"""Distributions."""
from pqueens.utils.import_utils import get_module_class


def from_config_create_distribution(distribution_options):
    """Create distribution object from distribution options dictionary.

    Args:
        distribution_options (dict): Dictionary with distribution description

    Returns:
        distribution: Distribution object
    """
    valid_types = {
        'normal': [".normal", "NormalDistribution"],
        'uniform': [".uniform", "UniformDistribution"],
        'lognormal': [".lognormal", "LogNormalDistribution"],
        'beta': [".beta", "BetaDistribution"],
    }

    distribution_type = distribution_options.get("distribution")
    distribution_class = get_module_class(distribution_options, valid_types, distribution_type)

    return distribution_class.from_config_create_distribution(distribution_options)
