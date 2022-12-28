"""Collection of utility functions and classes for PyMC."""


import numpy as np
import pymc as pm
import pytensor.tensor as pt

from pqueens.distributions import beta, exponential, lognormal, normal, uniform


class PymcDistributionWrapperWithGradient(pt.Op):
    """Op class for Data conversion.

    This PymcDistributionWrapperWithGradient  class is a wrapper for PyMC Distributions in QUEENS.

    Attributes:
        logpdf (fun): The log-pdf function
        logpdf_gradients (fun): The function to evaluate the gradient of the log-pdf
        logpdf_grad (obj): Wrapper for the gradient function of the log-pdf
    """

    itypes = [pt.dmatrix]  # input type
    otypes = [pt.dvector]  # output type

    def __init__(self, logpdf, logpdf_gradients):
        """Initzialise the wrapper for the functions.

        Args:
        logpdf (fun): The log-pdf function
        logpdf_gradients (fun): The function to evaluate the gradient of the pdf
        """
        self.logpdf = logpdf
        self.logpdf_gradients = logpdf_gradients
        # initialise the gradient Op
        self.logpdf_grad = PymcGradientWrapper(self.logpdf_gradients)

    # pylint: disable-next=unused-argument
    def perform(self, node, inputs, outputs):
        """Evaluate the pdf with this call."""
        (sample,) = inputs

        value = self.logpdf(sample)
        outputs[0][0] = np.array(value)

    def grad(self, inputs, g):
        """Jacobian product of the gradient."""
        (sample,) = inputs
        return [g[0] * self.logpdf_grad(sample)]


class PymcGradientWrapper(pt.Op):
    """Op class for Data conversion.

    This Class is a wrapper for the gradient of the distributions in QUEENS.

    Attributes:
        gradient_func (fun): The function to evaluate the gradient of the pdf
    """

    itypes = [pt.dmatrix]
    otypes = [pt.dmatrix]

    def __init__(self, gradient_func):
        """Initzialise the wrapper for the functions.

        Args:
        gradient_func (fun): The function to evaluate the gradient of the pdf
        """
        self.gradient_func = gradient_func

    # pylint: disable-next=unused-argument
    def perform(self, node, inputs, outputs):
        """Evaluate the gradient at the given sample."""
        (sample,) = inputs
        grads = self.gradient_func(sample)
        outputs[0][0] = grads


class PymcDistributionWrapper(pt.Op):
    """Op class for Data conversion.

    This PymcDistributionWrapper  class is a wrapper for PyMC Distributions in QUEENS.

    Attributes:
        logpdf (fun): The log-pdf function
    """

    itypes = [pt.dmatrix]  # input type
    otypes = [pt.dvector]  # output type

    def __init__(self, logpdf):
        """Initzialise the wrapper for the functions.

        Args:
        logpdf (fun): The log-pdf function
        """
        self.logpdf = logpdf

    # pylint: disable-next=unused-argument
    def perform(self, node, inputs, outputs):
        """Evaluate the pdf with this call."""
        (sample,) = inputs
        value = self.logpdf(sample)
        outputs[0][0] = np.array(value)


def from_config_create_pymc_distribution_dict(parameters, explicit_shape):
    """Get random variables in pymc distribution format.

    Args:
        parameters (obj): Parameters object

        explicit_shape (int): Explicit shape parameter for distribution dimension

    Returns:
        pymc distribution list
    """
    pymc_distribution_list = []

    # loop over rvs and create list
    for name, parameter in parameters.dict.items():
        pymc_distribution_list.append(
            from_config_create_pymc_distribution(parameter.distribution, name, explicit_shape)
        )
    # Pass the distribution list as arguments
    return pymc_distribution_list


def from_config_create_pymc_distribution(distribution, name, explicit_shape):
    """Create PyMC distribution object from queens distribution.

    Args:
        distribution (obj): Queens distribution object

        name (str): name of random variable
        explicit_shape (int): Explicit shape parameter for distribution dimension

    Returns:
        rv:     Random variable, distribution object in pymc format
    """
    shape = (explicit_shape, distribution.dimension)
    if isinstance(distribution, normal.NormalDistribution):
        rv = pm.MvNormal(
            name,
            mu=distribution.mean,
            cov=distribution.covariance,
            shape=shape,
        )
    elif isinstance(distribution, uniform.UniformDistribution):
        if np.all(distribution.lower_bound == 0):
            rv = pm.Uniform(
                name,
                lower=0,
                upper=distribution.upper_bound,
                shape=shape,
            )

        elif np.all(distribution.upper_bound == 0):
            rv = pm.Uniform(
                name,
                lower=distribution.lower_bound,
                upper=0,
                shape=shape,
            )
        else:
            rv = pm.Uniform(
                name,
                lower=distribution.lower_bound,
                upper=distribution.upper_bound,
                shape=shape,
            )
    elif isinstance(distribution, lognormal.LogNormalDistribution):
        if distribution.dimension == 1:
            std = distribution.covariance[0, 0] ** (1 / 2)
        else:
            raise NotImplementedError("Only 1D lognormals supported")

        rv = pm.LogNormal(
            name,
            mu=distribution.mean,
            sigma=std,
            shape=shape,
        )
    elif isinstance(distribution, exponential.ExponentialDistribution):
        rv = pm.Exponential(
            name,
            lam=distribution.rate,
            shape=shape,
        )
    elif isinstance(distribution, beta.BetaDistribution):
        rv = pm.Beta(
            name,
            alpha=distribution.a,
            beta=distribution.b,
            shape=shape,
        )
    else:
        raise NotImplementedError("Not supported distriubtion by QUEENS and/or PyMC")
    return rv
