"""
Weighted mixture of 2 multivariate Gaussian distributions.
"""

import numpy as np

from pqueens.utils import mcmc_utils

dim = 4

mean1 = 0.5 * np.ones(dim)
mean2 = -mean1

std = 0.1
cov = (std ** 2) * np.eye(dim)

weight1 = 0.1
weight2 = (1 - weight1)

gaussian1 = mcmc_utils.NormalProposal(mean=mean1, covariance=cov)
gaussian2 = mcmc_utils.NormalProposal(mean=mean2, covariance=cov)

def gaussian_mixture_logpdf(x1, x2, x3, x4):
    """ Multivariate Gaussian Mixture likelihood model

    Used as a basic test function for MCMC and SMC methods.

    The log likelihood is defined as (see [1]):

    :math:`f({x}) =\\log( w_1 \\frac{1}{(\\sqrt((2 \\pi)^k |\\Sigma_1|)}\\exp[-\\frac{1}{2}(x-\\mu_1)^T\\Sigma_1^{-1}(x-\\mu_1)]+ w_2 \\frac{1}{(\\sqrt((2 \\pi)^k |\\Sigma_2|)}\\exp[-\\frac{1}{2}(x-\\mu_2)^T\\Sigma_2^{-1}(x-\\mu_2)])`

    Args:
        x1 (float):
        x2 (float):
        x3 (float):
        x4 (float):

    Returns:
        numpy.array : The logpdf evaluated at x


    References:

        [1] https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    """
    x = np.array([x1, x2, x3, x4])

    y = np.log(weight1 * gaussian1.pdf(x) + weight2 * gaussian2.pdf(x))
    return y


def main(job_id, params):
    """ Interface to Gaussian Mixture logpdf model

    Args:
        job_id (int):  ID of job
        params (dict): Dictionary with parameters
    Returns:
        float: Value of logpdf of Gaussian Mixture at parameters specified in input dict
    """
    return gaussian_mixture_logpdf(params['x1'], params['x2'], params['x3'], params['x4'])


if __name__ == "__main__":
    result = np.exp(gaussian_mixture_logpdf(0.5, 0.5, 0.5, 0.5))
    print(f"results={result}")
    result = np.exp(gaussian_mixture_logpdf(-0.5, -0.5, -0.5, -0.5))
    print(f"results={result}")
