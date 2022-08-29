"""Distributions."""


def from_config_create_distribution(distribution_options):
    """Create distribution object from distribution options dictionary.

    Args:
        distribution_options (dict): Dictionary with distribution description

    Returns:
        distribution: Distribution object
    """
    from pqueens.distributions.beta import BetaDistribution
    from pqueens.distributions.exponential import ExponentialDistribution
    from pqueens.distributions.lognormal import LogNormalDistribution
    from pqueens.distributions.normal import NormalDistribution
    from pqueens.distributions.uniform import UniformDistribution
    from pqueens.utils.import_utils import get_module_attribute
    from pqueens.utils.valid_options_utils import get_option

    valid_options = {
        'normal': NormalDistribution,
        'uniform': UniformDistribution,
        'lognormal': LogNormalDistribution,
        'beta': BetaDistribution,
        'exponential': ExponentialDistribution,
    }

    if distribution_options.get("external_python_module"):
        module_path = distribution_options["external_python_module"]
        module_attribute = distribution_options.get("distribution")
        distribution_class = get_module_attribute(module_path, module_attribute)
    else:
        distribution_class = get_option(
            valid_options,
            distribution_options.get("distribution"),
            error_message="Requested distribution type not supported.",
        )
    return distribution_class.from_config_create_distribution(distribution_options)
