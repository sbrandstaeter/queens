"""Distributions."""

from pqueens.utils.valid_options_utils import get_option


def from_config_create_distribution(distribution_options):
    """Create distribution object from distribution options dictionary.

    Args:
        distribution_options (dict): Dictionary with distribution description

    Returns:
        distribution: Distribution object
    """
    from .beta import BetaDistribution
    from .lognormal import LogNormalDistribution
    from .normal import NormalDistribution
    from .uniform import UniformDistribution

    valid_options = {
        'normal': NormalDistribution,
        'uniform': UniformDistribution,
        'lognormal': LogNormalDistribution,
        'beta': BetaDistribution,
    }

    distribution_type = distribution_options.get('distribution', None)

    # TODO: This if statement might be unnecessary once the variable class is updated
    if distribution_type is not None:
        distribution_class = get_option(
            valid_options,
            distribution_type,
            error_message="Requested distribution type not supported.",
        )
        return distribution_class.from_config_create_distribution(distribution_options)
