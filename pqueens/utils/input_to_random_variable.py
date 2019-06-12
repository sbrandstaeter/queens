import numpy as np
import scipy.stats

# TODO write tests

def get_random_samples(description, num_samples):
    """ Get random samples based QUEENS description of distribution using numpy

    Args:
        description (dict):         Dictionary containing QUEENS distribution
                                    description
        num_samples (int):          Number of samples to generate

    Returns:
        np.array:                   Array with samples
    """

    random_number_generator = getattr(np.random, description['distribution'])
    my_args = list(description["distribution_parameter"])
    my_args.extend([num_samples])
    samples = random_number_generator(*my_args)
    return samples


def get_distribution_object(description):
    """ Get frozen scipy.stats.rv_frozen object based on QUEENS description of distribution

    Args:
        description (dict):         Dictionary containing QUEENS distribution
                                    description

    Returns:
        scipy.stats.rv_frozen:      Frozen scipy stats continuous random variable object
    """
    distribution_type = description.get('distribution', None)
    if distribution_type is None:
        distribution = None
    else:
        distribution_parameter = list(description.get('distribution_parameter',{}))
        if distribution_type == 'normal':
            distribution = scipy.stats.norm(distribution_parameter[0], distribution_parameter[1])
        elif distribution_type == 'lognormal':
            distribution = scipy.stats.lognorm(distribution_parameter[0], distribution_parameter[1])
        elif distribution_type == 'uniform':
            distribution = scipy.stats.uniform(distribution_parameter[0], distribution_parameter[1])
        else:
            raise ValueError("QUEENS can currently only deal with normal, lognormal"
                             " and uniform distributions, please fix your config file")
    return distribution
