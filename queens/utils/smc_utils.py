"""Collection of utility functions and classes for SMC algorithms."""

import math

import numpy as np
from particles import smc_samplers as ssp


def temper_logpdf_bayes(log_prior, log_like, tempering_parameter=1.0):
    """Bayesian tempering function.

    It phases from the prior to the posterior = like * prior.
    Special cases are:

    * tempering parameter = 0.0:
          We interpret this as "disregard contribution of the likelihood".
          Therefore, return just *log_prior*.

    * log_pior or log_like = `+inf`:
          Prohibit this case.
          The reasoning is that (`+inf` + `-inf`) is ambiguous.
          We know that `-inf` is likely to occur, e.g. in uniform priors.
          On the other hand, `+inf` is rather unlikely to be a reasonable
          value. Therefore, we chose to exclude it here.

    Args:
        log_prior (np.array): Array containing the values of the log-prior distribution
                              at sample points
        log_like (np.array): Array containing the values of the log-likelihood at
                             sample points
        tempering_parameter (float): Tempering parameter for resampling
    """
    # if either logpdf is positive infinite throw an error
    if np.isposinf(log_prior).any() or np.isposinf(log_like).any():
        raise ValueError("Positive infinite logpdf not supported.")

    # if the tempering_parameter is close to 0.0 return prior
    if math.isclose(tempering_parameter, 0.0, abs_tol=1e-8):
        return log_prior

    return tempering_parameter * log_like + log_prior


def temper_logpdf_generic(logpdf0, logpdf1, tempering_parameter=1.0):
    """Generic tempering function.

    It phases from one distribution (*pdf0*) to another (*pdf1*).

    Initial distribution: *pdf0*.

    Goal distribution: *pdf1*.

    * tempering parameter = 0.0:
        We interpret this as "disregard contribution of the goal pdf".
        Therefore, return *logpdf0*.

    * tempering parameter = 1.0:
        We interpret this as "we are fully transitioned." Therefore,
        ignore the contribution of the initial distribution.
        Therefore, return *logpdf1*.

    * logpdf0 or logpdf1 = `+inf`:
        Prohibit this case.
        The reasoning is that (`+inf` + `-inf`) is ambiguous.
        We know that `-inf` is likely to occur, e.g., in uniform
        distributions. On the other hand, `+inf` is rather unlikely to be
        a reasonable value. Therefore, we chose to exclude it here.

    Args:
        logpdf0: TODO_doc
        logpdf1: TODO_doc
        tempering_parameter: TODO_doc
    Returns:
        TODO_doc
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
    """Switch type of tempering function.

    Return the respective tempering function.

    Args:
        temper_type: TODO_doc
    Returns:
        TODO_doc
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
    """Calculate Effective Sample Size from current weights.

    We use the exp-log trick here to avoid numerical problems.

    Args:
        weights: TODO_doc
    Returns:
        ess: TODO_doc
    """
    ess = np.exp(np.log(np.sum(weights) ** 2) - np.log(np.sum(weights**2)))
    return ess


class StaticStateSpaceModel(ssp.StaticModel):
    """Model needed for the particles library implementation of SMC.

    Attributes:
        likelihood_model (object): Log-likelihood function.
        n_sims (int): Number of model calls.
    """

    def __init__(self, likelihood_model, data=None, prior=None):
        """Initialize Static State Space model.

        Args:
            likelihood_model (obj): Model for the log-likelihood function.
            data (np.array, optional): Optional data to define state space model.
                                       Defaults to None.
            prior (obj, optional): Model for the prior distribution. Defaults to None.
        """
        # Data is always set to `Ǹone` as we let QUEENS handle the actual likelihood computation
        super().__init__(data=data, prior=prior)
        self.likelihood_model = likelihood_model
        self.n_sims = 0

    def logpyt(self, theta, t):
        """Log-likelihood of Y_t, given parameter and previous datapoints.

        Args:
            theta (dict-like): theta['par'] is a ndarray containing the N values for parameter par
            t (int): time
        """
        raise NotImplementedError("StaticModel: logpyt not implemented")

    def loglik(self, theta, t=None):
        """Log. Likelihood function for *particles* SMC implementation.

        Args:
            theta (obj): Samples at which to evaluate the likelihood
            t (int): time (if set to None, the full log-likelihood is returned)

        Returns:
            The log likelihood
        """
        x = self.particles_array_to_numpy(theta)
        # Increase the model counter
        self.n_sims += len(x)
        return self.likelihood_model(x).flatten()

    def particles_array_to_numpy(self, theta):
        """Convert particles objects to numpy arrays.

        The *particles* library uses np.ndarrays with homemade variable dtypes.
        We need to convert this into numpy array to work with queens.

        Args:
            theta (np.ndarray with homemade dtype): *Particle* variables object

        Returns:
            np.ndarray: Numpy array of the particles
        """
        return np.lib.recfunctions.structured_to_unstructured(theta)

    def numpy_to_particles_array(self, samples):
        """Convert numpy arrays to particles objects.

        The *particles* library uses np.ndarrays with homemade variable dtypes.
        This method converts it back to the particles library type.

        Args:
            samples (np.ndarray): Numpy array samples

        Returns:
            np.ndarray with homemade dtype: *Particle* variables object
        """
        return samples.astype(self.prior.dtype)
