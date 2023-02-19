"""Collection of utility functions and classes for PyMC."""


import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.blocking import RaveledVars
from pymc.pytensorf import compile_pymc, floatX, join_nonshared_inputs
from pymc.step_methods.metropolis import tune

from pqueens.distributions import beta, exponential, lognormal, normal, uniform


class PymcDistributionWrapper(pt.Op):
    """Op class for Data conversion.

    This PymcDistributionWrapper  class is a wrapper for PyMC Distributions in QUEENS.

    Attributes:
        logpdf (fun): The log-pdf function
        logpdf_gradients (fun): The function to evaluate the gradient of the log-pdf
        logpdf_grad (obj): Wrapper for the gradient function of the log-pdf
    """

    itypes = [pt.dmatrix]
    otypes = [pt.dvector]

    def __init__(self, logpdf, logpdf_gradients=None):
        """Initzialise the wrapper for the functions.

        Args:
        logpdf (fun): The log-pdf function
        logpdf_gradients (fun): The function to evaluate the gradient of the pdf
        """
        self.logpdf = logpdf
        self.logpdf_gradients = logpdf_gradients
        self.logpdf_grad = PymcGradientWrapper(self.logpdf_gradients)

    # pylint: disable-next=unused-argument
    def perform(self, _node, inputs, outputs):
        """Call outside pdf function."""
        (sample,) = inputs

        value = self.logpdf(sample)
        outputs[0][0] = np.array(value)

    def grad(self, inputs, g):
        """Get gradient and multiply with upstream gradient."""
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
    def perform(self, _node, inputs, outputs):
        """Evaluate the gradient."""
        (sample,) = inputs
        if self.gradient_func is not None:
            grads = self.gradient_func(sample)
            outputs[0][0] = grads
        else:
            raise TypeError("Gradient function is not callable")


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
    for name, parameter in zip(parameters.names, parameters.to_list()):
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


def _metropolis_astep(self, sample):
    """Metropolis Sampler for multiple chains."""
    point_map_info = sample.point_map_info
    current_sample = sample.data
    if not self.steps_until_tune and self.tune:
        # Tune scaling parameter
        self.scaling = tune(self.scaling, self.accepted_sum / float(self.tune_interval))
        # Reset counter
        self.steps_until_tune = self.tune_interval
        self.accepted_sum[:] = 0

    delta = self.proposal_dist() * self.scaling
    proposed_sample = floatX(current_sample + delta)

    # Missuse mode to safe old posterior
    if self.mode is None:
        self.mode = np.sum(np.array(self.delta_logp(current_sample)), axis=0)

    logp_proposed = np.sum(np.array(self.delta_logp(proposed_sample)), axis=0)

    accept_rate = logp_proposed - self.mode

    new_sample, accepted = hastings_acceptance(accept_rate, proposed_sample, current_sample)
    self.mode[accepted] = logp_proposed[accepted]

    self.accept_rate_iter = accept_rate
    self.accepted_iter = accepted
    self.accepted_sum += accepted

    self.steps_until_tune -= 1

    stats = {
        "tune": self.tune,
        "scaling": np.mean(self.scaling),
        "accept": np.mean(np.exp(self.accept_rate_iter)),
        "accepted": np.mean(self.accepted_iter),
    }
    return RaveledVars(new_sample, point_map_info), [stats]


def hastings_acceptance(accept_rate, proposed_sample, current_sample):
    """Metropolis acceptance step.

    Args:
        accept_rate (np.array): Acceptance rate of samples
        proposed_sample (np.array): New sample
        current_sample (np.array): Old sample

    Returns:
        selected_sample (np.array): New sample for chain
        accept (np.array): Bool vector with acceptance/rejection
    """
    num_chains = len(accept_rate)
    parameter_dim = int(len(proposed_sample) / num_chains)

    accept = (
        np.log(
            np.random.uniform(
                size=num_chains,
            )
        )
        < accept_rate
    )

    bool_idx = np.kron(
        np.array(accept).astype(int),
        np.ones(shape=(parameter_dim,)),
    ).astype(bool)

    selected_samples = np.where(bool_idx, proposed_sample, current_sample)
    return selected_samples, accept


def logp(model):
    """Compilation function of likelihood.

    Args:
        model (obj): PyMC Model

    Returns:
        posterior_logp (function): Function to evaluate posterior
    """
    point = model.initial_point()
    model_vars = model.value_vars
    shared = pm.make_shared_replacements(point, model_vars, model)
    [logp0], inarray0 = join_nonshared_inputs(
        point=point, outputs=[model.logp(sum=False)], inputs=model_vars, shared_inputs=shared
    )

    posterior_logp = compile_pymc([inarray0], logp0)
    posterior_logp.trust_input = True
    return posterior_logp
