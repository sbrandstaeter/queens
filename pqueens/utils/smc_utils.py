"""
collection of utility functions and classes for Sequential Monte Carlo
(SMC) algorithms.
"""
import math

import numpy as np


def temper_logpdf_bayes(log_prior, log_like, tempering_parameter=1.0):
    """
    Bayesian tempering function.

    It phases from the prior to the posterior = like * prior.
    Special cases are:
    tempering parameter = 0.0:
        We interpret this as "disregard contribution of the likelihood".
        Therefore, return just log_prior.

    log_pior or log_like = +inf:
        Prohibit this case.
        The reasoning is that (+inf + -inf) is ambiguous.
        We know that -inf is likely to occur, e.g., in uniform priors.
        On the other hand, +inf is rather unlikely to be a reasonable
        value. Therefore, we chose to exclude it here.
     """
    # if either logpdf is positive infinite throw an error
    if np.isposinf(log_prior).any() or np.isposinf(log_like).any():
        raise ValueError("Positive infinite logpdf not supported.")

    # if the tempering_parameter is close to 0.0 return prior
    if math.isclose(tempering_parameter, 0.0, abs_tol=1e-8):
        return log_prior

    return tempering_parameter * log_like + log_prior


def temper_logpdf_generic(logpdf0, logpdf1, tempering_parameter=1.0):
    """
    Generic tempering function.

    It phases from one distribution (pdf0) to another (pdf1).
    initial distribution: pdf0
    goal distribution: pdf1

    tempering parameter = 0.0:
        We interpret this as "disregard contribution of the goal pdf".
        Therefore, return logpdf0.

    tempering parameter = 1.0:
        We interpret this as "we are fully transitioned." Therefore,
        ignore the contribution of the initial distribution.
        Therefore, return logpdf1.

    logpdf0 or logpdf1 = +inf:
        Prohibit this case.
        The reasoning is that (+inf + -inf) is ambiguous.
        We know that -inf is likely to occur, e.g., in uniform
        distributions. On the other hand, +inf is rather unlikely to be
        a reasonable value. Therefore, we chose to exclude it here.
    """
    # if either logpdf is positive infinite throw an error
    if np.isposinf(logpdf0).any() or np.isposinf(logpdf1).any():
        raise ValueError("Positive infinite logpdf not supported.")

    # if the tempering_parameter is close to 0.0 return initial logpdf
    if math.isclose(tempering_parameter, 0.0, abs_tol=1e-8):
        return logpdf0

    # if the tempering_parameter is close to 1.0 return final logpdf
    if math.isclose(tempering_parameter, 1.0):
        return logpdf1

    return (1.0 - tempering_parameter) * logpdf0 + tempering_parameter * logpdf1


def temper_factory(temper_type):
    """
    Switch type of tempering function.

    return the respective tempering function
    """

    if temper_type == 'bayes':
        return temper_logpdf_bayes
    if temper_type == 'generic':
        return temper_logpdf_generic

    valid_types = {'bayes', 'generic'}
    raise ValueError(
        f"Unknown type of tempering function: {temper_type}.\nValid choices are {valid_types}."
    )


def calc_ess(weights):
    """ Calculate Effective Sample Size from current weights. """

    ess = np.sum(weights) ** 2 / (np.sum(np.power(weights, 2)))
    return ess


from particles import smc_samplers as ssp


class StaticStateSpaceModel(ssp.StaticModel):
    """
    Model needed for the particles library implementation of SMC. 

    Attributes:
        prior (distribution StructDist object): `Particles` prior 
        likelihood_model (object): Log-likelihood function
        random_variable_keys (list): Lst containing the names of the RV
        n_sims (int): Number of model calls
    """

    def __init__(self, likelihood_model, random_variable_keys, data=None, prior=None):
        # Data is always set to `Ǹone` as we let QUEENS handle the actual likelihood computation
        super(StaticStateSpaceModel, self).__init__(data=None, prior=prior)
        self.likelihood_model = likelihood_model
        self.random_variable_keys = random_variable_keys
        self.n_sims = 0

    def loglik(self, theta):
        """
        Log. Likelihood function for `particles` SMC implementation

        Args:
            theta: Samples at which to evaluate the likehood

        Returns:
            The log likelihood
        
        """
        x = self.particles_array_to_numpy(theta)
        # Increase the model counter
        self.n_sims += len(x)
        return self.likelihood_model(x).flatten()

    def particles_array_to_numpy(self, theta):
        """
        The `particles` library uses an homemade variable type. We need to convert this into numpy 
        array to work with queens.

        Args:
            theta (`particles` object): `Particle` variables object 

        Returns:
            x (np.array): Numpy array from of the given data

        """
        x = None
        for theta_i in self.random_variable_keys:
            theta_i_np = theta[theta_i].flatten()
            if x is not None:
                x = np.vstack((x, theta_i_np))
            else:
                x = theta_i_np
        x = np.transpose(np.atleast_2d(x))
        return x
