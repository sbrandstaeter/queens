"""Distributions."""
from pqueens.utils.import_utils import get_module_class

valid_types = {
    'normal': ["pqueens.distributions.normal", "NormalDistribution"],
    'uniform': ["pqueens.distributions.uniform", "UniformDistribution"],
    'lognormal': ["pqueens.distributions.lognormal", "LogNormalDistribution"],
    'beta': ["pqueens.distributions.beta", "BetaDistribution"],
}


def from_config_create_distribution(distribution_options):
    """Create distribution object from distribution options dictionary.

    Args:
        distribution_options (dict): Dictionary with distribution description

    Returns:
        distribution: Distribution object
    """
    distribution_type = distribution_options.get("distribution")
    distribution_class = get_module_class(distribution_options, valid_types, distribution_type)
    distribution = distribution_class.from_config_create_distribution(distribution_options)
    return distribution
