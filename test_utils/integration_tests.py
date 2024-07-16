"""Utility methods used by the integration tests.."""

import numpy as np


def assert_monte_carlo_iterator_results(results, expected_mean, expected_var):
    """Assert the equality of the results with the expected values.

    Args:
        results (dict): Results dictionary from pickle file
        expected_mean (np.ndarray): Expected mean of the results
        expected_var (np.ndarray): Expected variance of the results
    """
    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["result"], expected_mean, decimal=4
    )
    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["variance"], expected_var, decimal=2
    )


def assert_sobol_index_iterator_results(results, expected_results):
    """Assert the equality of the results with the expected values.

    Args:
        results (dict): Results dictionary from pickle file
        expected_results (dict): Dictionary with expected results
    """
    np.testing.assert_allclose(results["sensitivity_indices"]["S1"], expected_results["S1"])
    np.testing.assert_allclose(
        results["sensitivity_indices"]["S1_conf"], expected_results["S1_conf"]
    )

    np.testing.assert_allclose(results["sensitivity_indices"]["ST"], expected_results["ST"])
    np.testing.assert_allclose(
        results["sensitivity_indices"]["ST_conf"], expected_results["ST_conf"]
    )

    np.testing.assert_allclose(results["sensitivity_indices"]["S2"], expected_results["S2"])
    np.testing.assert_allclose(
        results["sensitivity_indices"]["S2_conf"], expected_results["S2_conf"]
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
    mean = output["result"]
    variance = output["variance"]

    np.testing.assert_array_almost_equal(mean, mean_ref, decimal=decimals[0])
    np.testing.assert_array_almost_equal(variance, var_ref, decimal=decimals[1])

    if grad_mean_ref is not None:
        gradient_mean = output["grad_mean"]
        np.testing.assert_array_almost_equal(gradient_mean, grad_mean_ref, decimal=decimals[2])

    if grad_var_ref is not None:
        gradient_variance = output["grad_var"]
        np.testing.assert_array_almost_equal(gradient_variance, grad_var_ref, decimal=decimals[3])


def get_input_park91a(n_inputs):
    """Get inputs for the park91a benchmark function.

    Args:
        n_inputs (int): Number of inputs along x_1 and x_2

    Returns:
        np.ndarray, float, float: [x_1, x_2].T, x_3, x_4
    """
    x_3, x_4 = 0.5, 0.5
    x_1 = np.linspace(0.001, 0.999, n_inputs)
    x_2 = np.linspace(0.001, 0.999, n_inputs)
    xx_1, xx_2 = np.meshgrid(x_1, x_2)
    x_1_and_2 = np.vstack((xx_1.flatten(), xx_2.flatten())).T

    return x_1_and_2, x_3, x_4
