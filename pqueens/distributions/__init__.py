"""Distributions."""
from pqueens.utils.import_utils import get_module_class

VALID_TYPES = {
    'normal': ["pqueens.distributions.normal", "NormalDistribution"],
    'mean_field_normal': ["pqueens.distributions.mean_field_normal", "MeanFieldNormalDistribution"],
    'uniform': ["pqueens.distributions.uniform", "UniformDistribution"],
    'lognormal': ["pqueens.distributions.lognormal", "LogNormalDistribution"],
    'beta': ["pqueens.distributions.beta", "BetaDistribution"],
    'exponential': ['pqueens.distributions.exponential', 'ExponentialDistribution'],
    'free': ['pqueens.distributions.free', 'FreeVariable'],
}


def from_config_create_distribution(distribution_options):
    """Create distribution object from distribution options dictionary.

    Args:
        distribution_options (dict): Dictionary with distribution description

    Returns:
        distribution: Distribution object
    """
    distribution_class = get_module_class(distribution_options, VALID_TYPES, "type")
    distribution = distribution_class.from_config_create_distribution(distribution_options)
    return distribution
