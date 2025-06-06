#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
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
    """Perform generic tempering between two log-probability density functions.

    This function performs a linear interpolation between two log-probability density functions
    based on a tempering parameter. The tempering parameter determines the weight given to each
    log-probability density function in the transition from the initial distribution (*logpdf0*)
    to the goal distribution (*logpdf1*).

    The function handles the following scenarios:

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
        logpdf0 (float or np.array): Logarithm of the probability density function of the initial
                                     distribution.
        logpdf1 (float or np.array): Logarithm of the probability density function of the goal
                                     distribution.
        tempering_parameter (float): Parameter between 0 and 1 that controls the interpolation
                                     between `logpdf0` and `logpdf1`. A value of 0.0 corresponds to
                                     `logpdf0`, while a value of 1.0 corresponds to `logpdf1`.

    Returns:
        float or np.array: The tempered log-probability density function based on the
                              `tempering_parameter`.

    Raises:
        ValueError: If either `logpdf0` or `logpdf1` is positive infinity (`+inf`).
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
    """Return the appropriate tempering function based on the specified type.

    The tempering function can be used for transitioning between different log-probability density
    functions in various probabilistic models.

    Args:
        temper_type (str): Type of the tempering function to return. Valid options are:
            - "bayes": Returns the Bayes tempering function.
            - "generic": Returns the generic tempering function.

    Returns:
        function: The corresponding tempering function based on `temper_type`.

    Raises:
        ValueError: If `temper_type` is not one of the valid options ("bayes", "generic").
    """
    if temper_type == "bayes":
        return temper_logpdf_bayes
    if temper_type == "generic":
        return temper_logpdf_generic

    valid_types = {"bayes", "generic"}
    raise ValueError(
        f"Unknown type of tempering function: {temper_type}.\nValid choices are {valid_types}."
    )


def calc_ess(weights):
    """Calculate the Effective Sample Size (ESS) from the given weights.

    The Effective Sample Size (ESS) is a measure used to assess the quality of a set of weights
    by indicating how many independent samples would be required to achieve the same level of
    information as the current weighted samples. This is computed using the exp-log trick to
    improve numerical stability.

    Args:
        weights (np.array): An array of weights, typically representing the importance weights
                            of samples in a weighted sampling scheme.

    Returns:
        float: The Effective Sample Size (ESS) as calculated from the provided weights.
    """
    ess = np.exp(np.log(np.sum(weights) ** 2) - np.log(np.sum(weights**2)))
    return ess


class StaticStateSpaceModel(ssp.StaticModel):
    """Model needed for the particles library implementation of SMC.

    Attributes:
        likelihood_model (object): Log-likelihood function.
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

    def logpyt(self, theta, t):
        """Log-likelihood of Y_t, given parameter and previous datapoints.

        Args:
            theta (dict-like): theta['par'] is a ndarray containing the N values for parameter par
            t (int): time
        """
        raise NotImplementedError("StaticModel: logpyt not implemented")

    def loglik(self, theta, t=None):  # pylint: disable=unused-argument
        """Log. Likelihood function for *particles* SMC implementation.

        Args:
            theta (obj): Samples at which to evaluate the likelihood
            t (int): time (if set to None, the full log-likelihood is returned)

        Returns:
            The log likelihood
        """
        x = self.particles_array_to_numpy(theta)
        # Increase the model counter
        return self.likelihood_model(x).flatten()

    def particles_array_to_numpy(self, theta):
        """Convert particles objects to numpy arrays.

        The *particles* library uses np.ndarrays with homemade variable dtypes.
        We need to convert this into numpy arrays to work with queens.

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
            samples (np.ndarray): Samples

        Returns:
            np.ndarray with homemade dtype: *Particle* variables object
        """
        return samples.astype(self.prior.dtype)
