"""Distributions."""


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

    distribution_dict = {
        'normal': NormalDistribution,
        'uniform': UniformDistribution,
        'lognormal': LogNormalDistribution,
        'beta': BetaDistribution,
    }

    distribution_type = distribution_options.get('distribution', None)

    if distribution_type is not None:
        distribution_class = distribution_dict.get(distribution_type, None)

        if distribution_class is None:
            raise ValueError(
                "Requested distribution type not supported: {}.\n"
                "Supported types of distributions:  {}. "
                "".format(distribution_type, distribution_dict.keys())
            )

        return distribution_class.from_config_create_distribution(distribution_options)

    else:
        return None
