"""Weighted mixture of 2 multivariate Gaussian distributions.

Example adapted from from [1], section 3.4.

[1]: Minson, S. E., Simons, M. and Beck, J. L. (2013)      ‘Bayesian
inversion for finite fault earthquake source models I-theory and
algorithm’,      Geophysical Journal International, 194(3), pp.
1701–1726. doi: 10.1093/gji/ggt180.

**TODO_doc**: In this module, reference [1] is defined twice, maybe one
of them can be changed to [2]?
"""

import numpy as np

from queens.distributions.mixture import MixtureDistribution
from queens.distributions.normal import NormalDistribution

DIM = 4

MEAN_1 = 0.5 * np.ones(DIM)
MEAN_2 = -MEAN_1

STD = 0.1
COV = (STD**2) * np.eye(DIM)

WEIGHT_1 = 0.1
WEIGHT_2 = 1 - WEIGHT_1

GAUSSIAN_COMPONENT_1 = NormalDistribution(MEAN_1, COV)
GAUSSIAN_COMPONENT_2 = NormalDistribution(MEAN_2, COV)

GAUSSIAN_MIXTURE = MixtureDistribution(
    [WEIGHT_1, WEIGHT_2], [GAUSSIAN_COMPONENT_1, GAUSSIAN_COMPONENT_2]
)


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
    return GAUSSIAN_MIXTURE.logpdf(samples)


if __name__ == "__main__":
    result = np.exp(gaussian_mixture_4d_logpdf(np.array([0.5, 0.5, 0.5, 0.5]).reshape(1, -1)))
    result = np.exp(gaussian_mixture_4d_logpdf(np.array([-0.5, -0.5, -0.5, -0.5]).reshape(1, -1)))
