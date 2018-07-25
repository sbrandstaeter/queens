import numpy as np
import scipy.stats

# TODO write tests

def get_random_samples(description, num_samples):
    """ Get random samples based QUEENS discription of distribution using numpy

    Args:
        description (dict):         Dictionary containing QUEENS distribution
                                    description
        num_samples (int):          Number of samples to generate

    Returns:
        np.array:                   Array with samples
    """

    random_number_generator = getattr(np.random, description["distribution"])
    my_args = list(description["distribution_parameter"])
    my_args.extend([num_samples])
    samples = random_number_generator(*my_args)
    return samples


def get_distribution_object(description):
    """ Get frozen scipy.stats.rv object based on QUEENS discription of distribution

    Args:
        description (dict):         Dictionary containing QUEENS distribution
                                    description

    Returns:
        scipy.stats.rv:             Scipy stats continuous random variable object
    """
    distribution_parameter = list(description["distribution_parameter"])
    distribution = None
    if  description["distribution"] == 'normal':
        distribution = scipy.stats.norm(distribution_parameter)
    elif description["distribution"] == 'lognormal':
        distribution = scipy.stats.lognorm(distribution_parameter)
    elif description["distribution"] == 'uniform':
        distribution = scipy.stats.uniform(distribution_parameter)
    else:
        raise ValueError("QUEENS can currently only deal with normal, lognormal"
                         " and uniform distributions, please fix your config file")
    return distribution
