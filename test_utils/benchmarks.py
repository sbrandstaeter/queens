"""Utility methods used by the benchmark tests."""

import numpy as np

import queens.visualization.bmfia_visualization as qvis


def assert_weights_and_samples(results, expected_weights, expected_samples):
    """Assert the equality of some SMC results and the expected values.

    Args:
        results (dict): Results dictionary from pickle file
        expected_weights (np.array): Expected weights of the posterior samples. One weight for each
                                     sample row.
        expected_samples (np.array): Expected samples of the posterior. Each row is a different
                                     sample-vector. Different columns represent the different
                                     dimensions of the posterior.
    """
    samples = results['raw_output_data']['particles'].squeeze()
    weights = results['raw_output_data']['weights'].squeeze()

    dim_labels_lst = ['x_s', 'y_s']
    qvis.bmfia_visualization_instance.plot_posterior_from_samples(samples, weights, dim_labels_lst)

    np.testing.assert_array_almost_equal(weights, expected_weights, decimal=5)
    np.testing.assert_array_almost_equal(samples, expected_samples, decimal=5)
