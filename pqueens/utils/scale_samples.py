import warnings

import numpy as np
import scipy.stats

def scale_samples(samples, distribution_info):
    """ Scale samples that are in [0,1] range to other distributions

    Scale samples generated e.g. by a sobol sequence or latin hyper cube
    sampling and which thus lie within the [0,1]^d hypercube to samples arising
    from other distributions

    Args:
        samples (np.ndarray):     Array of dimensions num_params by N, where N
                                  is the number of samples

        distribution_info (list): list with distribution info, one for each
                                  parameter. Currently only uniform, normal, and
                                  lognormal distributions are supported. The
                                  parameters of the distributions are defined as
                                  usual:
                                    uniform:    lower and upper bounds
                                    normal:     mean and standard deviation
                                    lognormal:  location and scale

    Returns:
        np.ndarray:               Array with scaled samples with same size as
                                  input array
    """

    if samples.shape[1] != len(distribution_info):
        raise ValueError("Number of provided distributions must match number of \
            number of parameters ")

    # initializing array for scaled values
    scaled_samples = np.zeros_like(samples)

    # loop over the parameters
    for i in range(scaled_samples.shape[1]):
        # setting first and second arguments for distributions
        b1 = distribution_info[i]['distribution_parameter'][0]
        b2 = distribution_info[i]['distribution_parameter'][1]
        if 'min' in distribution_info[i] or 'max' in distribution_info[i]:
            warnings.warn("Bounds of parameters are ignored when using \
                scale_samples")


        if distribution_info[i]['distribution'] == 'uniform':
            if b1 >= b2:
                raise ValueError('''Uniform distribution: lower bound
                    must be less than upper bound''')
            else:
                scaled_samples[:, i] = samples[:, i] * (b2 - b1) + b1

        elif distribution_info[i]['distribution'] == 'normal':
            if b2 <= 0:
                raise ValueError('''Normal distribution: sigma must be > 0''')
            else:
                # in queens normal distributions are parameterized with mean and var
                # in salib normal distributions are parameterized via mean and std
                # -> we need to reparamteterize normal distributions
                b2 = np.sqrt(b2)
                scaled_samples[:, i] = scipy.stats.norm.ppf(
                    samples[:, i], loc=b1, scale=b2)

        elif distribution_info[i]['distribution'] == 'lognormal':
            if b2 <= 0:
                raise ValueError(
                    '''Lognormal distribution: scale must be > 0''')
            else:
                scaled_samples[:, i] = np.exp(
                    scipy.stats.norm.ppf(samples[:, i], loc=b1, scale=b2))

        else:
            valid_distributions = ['uniform', 'normal', 'lognormal']
            raise ValueError('Distributions: choose one of %s' %
                             ", ".join(valid_distributions))

    return scaled_samples
