"""Test cases for Sobol index estimation with metamodel uncertainty."""

import numpy as np

from queens.main import run
from queens.utils.io_utils import load_result


def test_sobol_indices_ishigami_gp_uncertainty(inputdir, tmp_path):
    """Test case for Sobol indices based on GP realizations."""
    run(inputdir / 'sobol_indices_ishigami_gp_uncertainty.yml', tmp_path)

    results = load_result(tmp_path / 'xxx.pickle')

    expected_s1 = np.array(
        [
            [0.30469190, 0.00014149, 0.00005653, 0.00016402, 0.02331390, 0.01473639, 0.02510155],
            [0.38996188, 0.00039567, 0.00049108, 0.00003742, 0.03898644, 0.04343343, 0.01198891],
            [0.00383826, 0.00030052, 0.00008825, 0.00044747, 0.03397690, 0.01841250, 0.04146019],
        ]
    )
    expected_st = np.array(
        [
            [0.55816767, 0.00050181, 0.00001702, 0.00082728, 0.04390555, 0.00808476, 0.05637328],
            [0.50645929, 0.00022282, 0.00022212, 0.00010188, 0.02925636, 0.02921057, 0.01978344],
            [0.30344671, 0.00010415, 0.00011769, 0.00004659, 0.02000237, 0.02126261, 0.01337864],
        ]
    )
    expected_s2 = np.array(
        [
            [0.00461299, 0.00215561, 0.00006615, 0.00352044, 0.09099820, 0.01594134, 0.11629120],
            [0.19526686, 0.00147909, 0.00059668, 0.00169727, 0.07537822, 0.04787620, 0.08074639],
            [0.06760761, 0.00004854, 0.00002833, 0.00007491, 0.01365552, 0.01043203, 0.01696364],
        ]
    )

    np.testing.assert_allclose(results['first_order'].values, expected_s1, atol=1e-05)
    np.testing.assert_allclose(results['second_order'].values, expected_s2, atol=1e-05)
    np.testing.assert_allclose(results['total_order'].values, expected_st, atol=1e-05)


def test_sobol_indices_ishigami_gp_uncertainty_third_order(inputdir, tmp_path):
    """Test case for third-order Sobol indices."""
    run(inputdir / 'sobol_indices_ishigami_gp_uncertainty_third_order.yml', tmp_path)

    results = load_result(tmp_path / 'xxx.pickle')

    expected_s3 = np.array(
        [[0.23426643, 0.00801287, 0.00230968, 0.00729179, 0.17544544, 0.09419407, 0.16736517]]
    )

    np.testing.assert_allclose(results['third_order'].values, expected_s3, atol=1e-05)


def test_sobol_indices_ishigami_gp_mean(inputdir, tmp_path):
    """Test case for Sobol indices based on GP mean."""
    run(inputdir / 'sobol_indices_ishigami_gp_mean.yml', tmp_path)

    results = load_result(tmp_path / 'xxx.pickle')

    expected_s1 = np.array(
        [
            [0.28879163, 0.00022986, np.nan, 0.00022986, 0.02971550, np.nan, 0.02971550],
            [0.45303182, 0.00000033, np.nan, 0.00000033, 0.00112608, np.nan, 0.00112608],
            [0.07601656, 0.00000084, np.nan, 0.00000084, 0.00179415, np.nan, 0.00179415],
        ]
    )
    expected_st = np.array(
        [
            [0.47333086, 0.00093263, np.nan, 0.00093263, 0.05985535, np.nan, 0.05985535],
            [0.48403078, 0.00000185, np.nan, 0.00000185, 0.00266341, np.nan, 0.00266341],
            [0.23926036, 0.00003290, np.nan, 0.00003290, 0.01124253, np.nan, 0.01124253],
        ]
    )

    np.testing.assert_allclose(results['first_order'].values, expected_s1, atol=1e-05)
    np.testing.assert_allclose(results['total_order'].values, expected_st, atol=1e-05)
