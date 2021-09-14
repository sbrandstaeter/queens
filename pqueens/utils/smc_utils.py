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
    """ 
    Calculate Effective Sample Size from current weights.
    We use the exp-log trick here to avoid numerical problems.
    """

    ess = np.exp(np.log(np.sum(weights) ** 2) - np.log(np.sum(weights ** 2)))
    return ess
