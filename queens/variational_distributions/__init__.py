"""Variational distributions init."""

from queens.distributions.particles import ParticleDiscreteDistribution
from queens.variational_distributions.full_rank_normal import FullRankNormalVariational
from queens.variational_distributions.joint import JointVariational
from queens.variational_distributions.mean_field_normal import MeanFieldNormalVariational
from queens.variational_distributions.mixture_model import MixtureModelVariational
from queens.variational_distributions.particle import ParticleVariational


def create_simple_distribution(distribution_options):
    """Create a simple variational distribution object.

    No nested distributions like mixture models.

    Args:
        distribution_options (dict): Dict for the distribution options

    Returns:
        distribution_obj: Variational distribution object
    """
    distribution_family = distribution_options.get('variational_family', None)
    if distribution_family == "normal":
        dimension = distribution_options.get('dimension')
        approximation_type = distribution_options.get('variational_approximation_type', None)
        distribution_obj = create_normal_distribution(dimension, approximation_type)
    elif distribution_family == "particles":
        dimension = distribution_options["dimension"]
        probabilities = distribution_options["probabilities"]
        sample_space = distribution_options["sample_space"]
        distribution_obj = ParticleDiscreteDistribution(probabilities, sample_space)

    return distribution_obj


def create_normal_distribution(dimension, approximation_type):
    """Create a normal variational distribution object.

    Args:
        dimension (int): Dimension of latent variable
        approximation type (str): Full rank or mean field

    Returns:
        distribution_obj: Variational distribution object
    """
    if approximation_type == "mean_field":
        distribution_obj = MeanFieldNormalVariational(dimension)
    elif approximation_type == "fullrank":
        distribution_obj = FullRankNormalVariational(dimension)
    else:
        supported_types = {'mean_field', 'fullrank'}
        raise ValueError(
            f"Requested variational approximation type not supported: {approximation_type}.\n"
            f"Supported types:  {supported_types}. "
        )
    return distribution_obj


def create_mixture_model_distribution(base_distribution, dimension, n_components):
    """Create a mixture model variational distribution.

    Args:
        base_distribution: Variational distribution object
        dimension (int): Dimension of latent variable
        n_components (int): Number of mixture components

    Returns:
        distribution_obj: Variational distribution object
    """
    if n_components > 1:
        distribution_obj = MixtureModelVariational(base_distribution, dimension, n_components)
    else:
        raise ValueError(
            "Number of mixture components has be greater than 1. If a single component is"
            "desired use the respective variational distribution directly (is computationally"
            "more efficient)."
        )
    return distribution_obj


def create_variational_distribution(distribution_options):
    """Create variational distribution object from dictionary.

    Args:
        distribution_options (dict): Dictionary containing parameters
                                     defining the distribution

    Returns:
        distribution: Variational distribution object
    """
    distribution_family = distribution_options.get('variational_family', None)
    supported_simple_distribution_families = ['normal', 'particles']
    if distribution_family in supported_simple_distribution_families:
        distribution_obj = create_simple_distribution(distribution_options)
    elif distribution_family == "mixture_model":
        dimension = distribution_options.get('dimension')
        n_components = distribution_options.get('n_components')
        base_distribution_options = distribution_options.get('base_distribution')
        base_distribution_options.update({"dimension": dimension})
        base_distribution_obj = create_simple_distribution(base_distribution_options)
        distribution_obj = create_mixture_model_distribution(
            base_distribution_obj, dimension, n_components
        )
    elif distribution_family == "joint":
        dimension = distribution_options.get('dimension')
        distributions = []
        for distribution_config in distribution_options["distributions"]:
            distributions.append(create_variational_distribution(distribution_config))
        distribution_obj = JointVariational(distributions, dimension)
    else:
        supported_distributions = [
            "mixture_model",
            "joint",
        ] + supported_simple_distribution_families
        raise ValueError(
            f"Requested variational family type not supported: {distribution_family}.\n"
            f"Supported types:  {supported_distributions}."
        )
    return distribution_obj
