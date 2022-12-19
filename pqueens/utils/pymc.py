"""Collection of utility functions and classes for PyMC."""


import aesara.tensor as at
import numpy as np
import pymc as pm

from pqueens.distributions import beta, exponential, lognormal, normal, uniform


class PymcDistributionWrapper(at.Op):
    """Op class for Data conversion.

    This pymc_distribution_rapper class is a wrapper for PyMC Distributions in QUEENS.

    Attributes:
        loglike (fun): The log-pdf function
        pdf_gradients (fun): The function to evaluate the gradient of the pdf
    """

    itypes = [at.dmatrix]  # input type
    otypes = [at.dvector]  # output type

    def __init__(self, logpdf, pdf_gradients):
        """Initzialise the wrapper for the functions.

        Args:
        logpdf (fun): The log-pdf function
        pdf_gradients (fun): The function to evaluate the gradient of the pdf
        """
        self.logpdf = logpdf
        self.pdf_gradients = pdf_gradients
        # initialise the gradient Op
        self.logpdfgrad = PdfGrad(self.pdf_gradients)

    # pylint: disable-next=unused-argument
    def perform(self, node, inputs, outputs):
        """Evaluate the pdf with this call."""
        (sample,) = inputs

        value = self.logpdf(sample)
        outputs[0][0] = np.array(value)

    def grad(self, inputs, g):
        """Jacobian product of the gradient."""
        (sample,) = inputs
        return [g[0] * self.logpdfgrad(sample)]


class PdfGrad(at.Op):
    """Op class for Data conversion.

    This Class is a wrapper for the gradient of the distributions in QUEENS.

    Attributes:
        gradient_func (fun): The function to evaluate the gradient of the pdf
    """

    itypes = [at.dmatrix]
    otypes = [at.dmatrix]

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


class PymcDistributionWrapperWithoutGrad(at.Op):
    """Op class for Data conversion.

    This pymc_distribution_rapper_wo_grad class is a wrapper for PyMC Distributions in QUEENS.

    Attributes:
        loglike (fun): The log-pdf function
    """

    itypes = [at.dmatrix]  # input type
    otypes = [at.dvector]  # output type

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
        distribution:     Distribution object in pymc format
    """
    shape = (explicit_shape, distribution.dimension)
    if isinstance(distribution, normal.NormalDistribution):
        distribution = pm.MvNormal(
            name,
            mu=distribution.mean,
            cov=distribution.covariance,
            shape=shape,
        )
    elif isinstance(distribution, uniform.UniformDistribution):
        if np.all(distribution.lower_bound == distribution.lower_bound[0]) and np.all(
            distribution.upper_bound == distribution.upper_bound[0]
        ):
            distribution = pm.Uniform(
                name,
                lower=distribution.lower_bound[0],
                upper=distribution.upper_bound[0],
                shape=shape,
            )
        elif np.all(distribution.lower_bound == distribution.lower_bound[0]) and not np.all(
            distribution.upper_bound == distribution.upper_bound[0]
        ):
            distribution = pm.Uniform(
                name,
                lower=distribution.lower_bound[0],
                upper=distribution.upper_bound,
                shape=shape,
            )
        elif np.all(distribution.upper_bound == distribution.upper_bound[0]) and not np.all(
            distribution.lower_bound == distribution.lower_bound[0]
        ):
            distribution = pm.Uniform(
                name,
                lower=distribution.lower_bound,
                upper=distribution.upper_bound[0],
                shape=shape,
            )
        else:
            distribution = pm.Uniform(
                name,
                lower=distribution.lower_bound,
                upper=distribution.upper_bound,
                shape=shape,
            )
    elif isinstance(distribution, lognormal.LogNormalDistribution):
        if distribution.dimension == 1:
            std = distribution.covariance[0, 0] ** (1 / 2)
        elif distribution.dimension != 1 and np.all(np.array(distribution.covariance.shape) == 1):
            std = distribution.covariance[0, 0] ** (1 / 2)
        elif (
            np.count_nonzero(
                distribution.covariance - np.diag(np.diagonal(distribution.covariance))
            )
            == 0
        ):
            std = np.diagonal(distribution.covariance) ** (1 / 2)
        else:
            raise NotImplementedError("There is no multivariate LogNormal-Distribution in PyMC")
        
        distribution = pm.LogNormal(
            name,
            mu=distribution.mean,
            sigma=std,
            shape=shape,
        )
    elif isinstance(distribution, exponential.ExponentialDistribution):
        distribution = pm.Exponential(
            name,
            lam=distribution.rate,
            shape=shape,
        )
    elif isinstance(distribution, beta.BetaDistribution):
        distribution = pm.Beta(
            name,
            alpha=distribution.a,
            beta=distribution.b,
            shape=shape,
        )
    else:
        raise NotImplementedError("Not supported distriubtion by QUEENS and/or PyMC")
    return distribution
