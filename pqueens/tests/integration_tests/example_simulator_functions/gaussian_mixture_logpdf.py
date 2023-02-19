"""Weighted mixture of 2 multivariate Gaussian distributions.

Example adapted from from [1], section 3.4.

[1]: Minson, S. E., Simons, M. and Beck, J. L. (2013)
     ‘Bayesian inversion for finite fault earthquake source models I-theory and algorithm’,
     Geophysical Journal International, 194(3), pp. 1701–1726. doi: 10.1093/gji/ggt180.

**TODO_doc**: In this module, reference [1] is defined twice, maybe one of them can be changed to [2]?
"""
# pylint: disable=invalid-name


import numpy as np

from pqueens.distributions import from_config_create_distribution

dim = 4

mean1 = 0.5 * np.ones(dim)
mean2 = -mean1

std = 0.1
cov = (std**2) * np.eye(dim)

weight1 = 0.1
weight2 = 1 - weight1

dist_options_1 = {'type': 'normal', 'mean': mean1, 'covariance': cov}
dist_options_2 = {'type': 'normal', 'mean': mean2, 'covariance': cov}
gaussian_component_1 = from_config_create_distribution(dist_options_1)
gaussian_component_2 = from_config_create_distribution(dist_options_2)


def gaussian_mixture_4d_logpdf(samples):
    r"""Multivariate Gaussian Mixture likelihood model.

    Used as a basic test function for MCMC and SMC methods.

    The log likelihood is defined as (see [1]):

    :math:`f({x}) =\log \left( w_1 \frac{1}{\sqrt{(2 \pi)^k |\Sigma_1|}}
    \exp \left[-\frac{1}{2}(x-\mu_1)^T\Sigma_1^{-1}(x-\mu_1) \right]+ w_2
    \frac{1}{\sqrt{(2 \pi)^k |\Sigma_2|}}
    \exp \left[-\frac{1}{2}(x-\mu_2)^T\Sigma_2^{-1}(x-\mu_2) \right] \right)`

    Returns:
        numpy.array: The logpdf evaluated at *x*

    References:
        [1] https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    """
    y = np.log(
        weight1 * gaussian_component_1.pdf(samples) + weight2 * gaussian_component_2.pdf(samples)
    )
    return y


if __name__ == "__main__":
    result = np.exp(gaussian_mixture_4d_logpdf(np.array([0.5, 0.5, 0.5, 0.5]).reshape(1, -1)))
    result = np.exp(gaussian_mixture_4d_logpdf(np.array([-0.5, -0.5, -0.5, -0.5]).reshape(1, -1)))
