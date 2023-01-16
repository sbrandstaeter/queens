"""Collection of utils for Markov Chain Monte Carlo algorithms."""

import abc

import numpy as np


def mh_select(log_acceptance_probability, current_sample, proposed_sample):
    """Do Metropolis Hastings selection.

    Args:
        log_acceptance_probability: TODO_doc
        current_sample: TODO_doc
        proposed_sample: TODO_doc
    Returns:
        TODO_doc
    """
    isfinite = np.isfinite(log_acceptance_probability)
    accept = (
        np.log(np.random.uniform(size=log_acceptance_probability.shape))
        < log_acceptance_probability
    )

    bool_idx = isfinite * accept

    selected_samples = np.where(bool_idx, proposed_sample, current_sample)

    return selected_samples, bool_idx


def tune_scale_covariance(scale_covariance, accept_rate):
    r"""Tune the acceptance rate according to the last tuning interval.

    The goal is an acceptance rate within 20\% - 50\%.
    The (acceptance) rate is adapted according to the following rule:


        +------------------+-----------------------------+
        | Acceptance Rate  |  Variance adaptation factor |
        +==================+=============================+
        |     <0.001       |              x 0.1          |
        +------------------+-----------------------------+
        |      <0.05       |              x 0.5          |
        +------------------+-----------------------------+
        |      <0.2        |              x 0.9          |
        +------------------+-----------------------------+
        |      >0.5        |              x 1.1          |
        +------------------+-----------------------------+
        |      >0.75       |              x 2            |
        +------------------+-----------------------------+
        |      >0.95       |              x 10           |
        +------------------+-----------------------------+

    The implementation is modified from [1].

    Reference:
    [1]: https://github.com/pymc-devs/pymc3/blob/master/pymc3/step_methods/metropolis.py

    **TODO_doc:** The link is broken !

    Args:
        scale_covariance: TODO_doc
        accept_rate: TODO_doc
    Returns:
        scale_covariance: TODO_doc
    """
    scale_covariance = np.where(accept_rate < 0.001, scale_covariance * 0.1, scale_covariance)
    scale_covariance = np.where(
        (accept_rate >= 0.001) * (accept_rate < 0.05), scale_covariance * 0.5, scale_covariance
    )
    scale_covariance = np.where(
        (accept_rate >= 0.05) * (accept_rate < 0.2), scale_covariance * 0.9, scale_covariance
    )
    scale_covariance = np.where(
        (accept_rate > 0.5) * (accept_rate <= 0.75), scale_covariance * 1.1, scale_covariance
    )
    scale_covariance = np.where(
        (accept_rate > 0.75) * (accept_rate <= 0.95), scale_covariance * 2.0, scale_covariance
    )
    scale_covariance = np.where((accept_rate > 0.95), scale_covariance * 10.0, scale_covariance)

    return scale_covariance
