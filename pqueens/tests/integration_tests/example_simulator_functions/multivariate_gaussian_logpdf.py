import numpy as np

from pqueens.utils import mcmc_utils

dim = 2

meas_data = [0.0, 0.0]
cov = [[1.0, 0.5], [0.5, 1.0]]

A = np.eye(dim, dim)
b = np.zeros(dim)


gauss_like = mcmc_utils.NormalProposal(mean=meas_data, covariance=cov)


def gaussian_logpdf(x1, x2):
    """2D Gaussian likelihood model.

    Used as a basic test function for MCMC methods.

    The log likelihood is defined as (see [1]):

    :math:`f({x}) = -\\frac{1}{2}(x-\\mu)^T\\Sigma^{-1}(x-\\mu)-\\log(\\sqrt((2 \\pi)^k |\\Sigma|)`

    Args:
        x1 (float):
        x2 (float):

    Returns:
        numpy.array : The logpdf evaluated at x


    References:

        [1] https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    """

    x = np.array([x1, x2])

    model_data = np.dot(A, x) + b

    y = gauss_like.logpdf(model_data)
    return y


def main(job_id, params):
    """Interface to 1D Guassian model.

    Args:
        job_id (int):  ID of job
        params (dict): Dictionary with parameters
    Returns:
        float: Value of Gaussian at parameters specified in input dict
    """
    return gaussian_logpdf(params['x1'], params['x2'])
