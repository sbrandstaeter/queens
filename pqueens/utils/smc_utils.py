"""
collection of utility functions and classes for Sequential Monte Carlo
(SMC) algorithms.
"""

def temper_logpdf_bayes(log_prior, log_like, tempering_parameter=1.0):
    """
    Bayesian tempering function.

    It phases from the prior to the posterior = like * prior.
    """
    return tempering_parameter * log_like + log_prior

def temper_logpdf_generic(logpdf0, logpdf1, tempering_parameter=1.0):
    """
    Generic tempering function.

    It phases from one distribution to another.
    """
    return tempering_parameter * logpdf1 + logpdf0 * (1.0 - tempering_parameter)


def temper_factory(temper_type):
    """
    Switch type of tempering function.

    return the respective tempering function
    """

    if temper_type == 'bayes':
        return temper_logpdf_bayes
    elif temper_type == 'generic':
        return temper_logpdf_generic
    else:
        valid_types = {'bayes', 'generic'}
        raise ValueError(f"Unknown type of tempering function: {temper_type}."
                         f"\nValid choices are {valid_types}.")
