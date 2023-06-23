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
            [0.30625942, 0.00035531, 0.00035066, 0.00017291, 0.03694485, 0.03670208, 0.0257729],
            [0.40668483, 0.00009362, 0.00011696, 0.00000335, 0.01896413, 0.02119653, 0.00358803],
            [0.0033352, 0.00033158, 0.00020779, 0.00032785, 0.03568957, 0.02825252, 0.0354881],
        ]
    )
    expected_st = np.array(
        [
            [0.53240643, 0.00028998, 0.00007168, 0.00040097, 0.03337565, 0.0165943, 0.0392467],
            [0.5430871, 0.0000441, 0.0000359, 0.00004841, 0.01301616, 0.01174374, 0.01363711],
            [0.22465854, 0.00009897, 0.00012154, 0.00001866, 0.0194989, 0.02160803, 0.00846554],
        ]
    )
    expected_s2 = np.array(
        [
            [0.0370762, 0.00149522, 0.00039872, 0.00202231, 0.0757881, 0.03913635, 0.08813988],
            [0.15375281, 0.00179645, 0.00158001, 0.00122951, 0.08307221, 0.07790729, 0.06872497],
            [0.05834652, 0.00095342, 0.00106968, 0.0004029, 0.06051866, 0.06410251, 0.03934119],
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
