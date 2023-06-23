"""Test Sobol indices estimation with Gaussian process surrogate."""
import pickle

import numpy as np

from pqueens import run


def test_sobol_indices_ishigami_gp(inputdir, tmp_path):
    """Test Sobol indices estimation with Gaussian process surrogate."""
    run(inputdir / 'sobol_indices_ishigami_gp.yml', tmp_path)

    result_file = tmp_path / 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    expected_result_s1 = np.array([0.37365542, 0.49936914, -0.00039217])
    expected_result_s1_conf = np.array([0.14969221, 0.18936135, 0.0280309])

    np.testing.assert_allclose(results['sensitivity_indices']['S1'], expected_result_s1, atol=1e-05)
    np.testing.assert_allclose(
        results['sensitivity_indices']['S1_conf'], expected_result_s1_conf, atol=1e-05
    )
