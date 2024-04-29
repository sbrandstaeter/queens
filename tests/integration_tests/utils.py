"""Utility methods used by the integration tests.."""

import numpy as np


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
