"""Utility methods for the entire test suite."""

import numpy as np

from queens.example_simulator_functions.gaussian_logpdf import (
    gaussian_1d_logpdf,
    gaussian_2d_logpdf,
)


def assert_surrogate_model_output(
    output, mean_ref, var_ref, grad_mean_ref=None, grad_var_ref=None, decimals=(2, 2, 2, 2)
):
    """Assert the equality of the output with the provided reference values.

    Args:
        output (dict): surrogate model output
        mean_ref (np.ndarray): reference mean
        var_ref (np.ndarray): reference variance
        grad_mean_ref (np.ndarray): reference gradient of the mean
        grad_var_ref (np.ndarray): reference gradient of the variance
        decimals (lst): list of desired decimal precisions
    """
    mean = output['result']
    variance = output['variance']

    np.testing.assert_array_almost_equal(mean, mean_ref, decimal=decimals[0])
    np.testing.assert_array_almost_equal(variance, var_ref, decimal=decimals[1])

    if grad_mean_ref is not None:
        gradient_mean = output['grad_mean']
        np.testing.assert_array_almost_equal(gradient_mean, grad_mean_ref, decimal=decimals[2])

    if grad_var_ref is not None:
        gradient_variance = output['grad_var']
        np.testing.assert_array_almost_equal(gradient_variance, grad_var_ref, decimal=decimals[3])


def target_density_gaussian_1d(self, samples):  # pylint: disable=unused-argument
    """Target posterior density."""
    samples = np.atleast_2d(samples)
    log_likelihood = gaussian_1d_logpdf(samples).reshape(-1, 1)

    return log_likelihood


def target_density_gaussian_2d(self, samples):  # pylint: disable=unused-argument
    """Target likelihood density."""
    samples = np.atleast_2d(samples)
    log_likelihood = gaussian_2d_logpdf(samples).flatten()

    cov = [[1.0, 0.5], [0.5, 1.0]]
    cov_inverse = np.linalg.inv(cov)
    gradient = -np.dot(cov_inverse, samples.T).T

    return log_likelihood, gradient
