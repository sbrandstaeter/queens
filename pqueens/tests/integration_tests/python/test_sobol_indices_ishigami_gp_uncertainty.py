"""Test cases for Sobol index estimation with metamodel uncertainty."""
import pickle

import numpy as np

from pqueens import run


def test_sobol_indices_ishigami_gp_uncertainty(inputdir, tmp_path):
    """Test case for Sobol indices based on GP realizations."""
    run(inputdir / 'sobol_indices_ishigami_gp_uncertainty.yml', tmp_path)

    result_file = tmp_path / 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    expected_s1 = np.array(
        [
            [0.31350375, 0.00016172, 0.00004503, 0.00023026, 0.02492464, 0.01315197, 0.02974084],
            [0.40761867, 0.00023214, 0.00028962, 0.00002062, 0.02986206, 0.03335536, 0.00889987],
            [0.00260481, 0.00027457, 0.00012816, 0.00028993, 0.03247676, 0.02218802, 0.03337307],
        ]
    )
    expected_st = np.array(
        [
            [0.53156809, 0.00040527, 0.00020167, 0.00042787, 0.03945681, 0.02783366, 0.04054199],
            [0.53870666, 0.00018941, 0.00021056, 0.00004338, 0.02697435, 0.02844035, 0.01290956],
            [0.23157817, 0.00010889, 0.00012691, 0.00001245, 0.02045221, 0.02208020, 0.00691648],
        ]
    )
    expected_s2 = np.array(
        [
            [0.01643402, 0.00425524, 0.00350314, 0.00283827, 0.12785277, 0.11600508, 0.10441788],
            [0.13484543, 0.00117350, 0.00054347, 0.00135420, 0.06714138, 0.04569172, 0.07212553],
            [0.06452848, 0.00135066, 0.00144074, 0.00047729, 0.07203119, 0.07439452, 0.04281916],
        ]
    )

    np.testing.assert_allclose(results['first_order'].values, expected_s1, atol=1e-05)
    np.testing.assert_allclose(results['second_order'].values, expected_s2, atol=1e-05)
    np.testing.assert_allclose(results['total_order'].values, expected_st, atol=1e-05)


def test_sobol_indices_ishigami_gp_uncertainty_third_order(inputdir, tmp_path):
    """Test case for third-order Sobol indices."""
    run(inputdir / 'sobol_indices_ishigami_gp_uncertainty_third_order.yml', tmp_path)

    result_file = tmp_path / 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    expected_s3 = np.array([[0.22099, 0.02901, 0.02153, 0.01082, 0.33383, 0.28759, 0.20386]])

    # For third-order indices, we can only test for a very rough tolerance because:
    #   - the scipy optimizer used in GPy results in changes of hyperparameters in the twelfth
    #     decimal place
    #   - GP predictions change in the twelfth decimal place
    #   - first- and second-order results change in the tenth decimal place
    #   - all those changes drastically add up in the third order indices
    np.testing.assert_allclose(results['third_order'].values, expected_s3, atol=1e-01)


def test_sobol_indices_ishigami_gp_mean(inputdir, tmp_path):
    """Test case for Sobol indices based on GP mean."""
    run(inputdir / 'sobol_indices_ishigami_gp_mean.yml', tmp_path)

    result_file = tmp_path / "xxx.pickle"
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    expected_s1 = np.array(
        [
            [0.32790810, 0.00012229, np.nan, 0.00012229, 0.02167385, np.nan, 0.02167385],
            [0.47866688, 0.00008283, np.nan, 0.00008283, 0.01783824, np.nan, 0.01783824],
            [0.08099259, 0.00000252, np.nan, 0.00000252, 0.00311120, np.nan, 0.00311120],
        ]
    )
    expected_st = np.array(
        [
            [0.45362563, 0.00020541, np.nan, 0.00020541, 0.02809069, np.nan, 0.02809069],
            [0.53906090, 0.00004791, np.nan, 0.00004791, 0.01356595, np.nan, 0.01356595],
            [0.17905852, 0.00000004, np.nan, 0.00000004, 0.00040116, np.nan, 0.00040116],
        ]
    )

    np.testing.assert_allclose(results['first_order'].values, expected_s1, atol=1e-05)
    np.testing.assert_allclose(results['total_order'].values, expected_st, atol=1e-05)
