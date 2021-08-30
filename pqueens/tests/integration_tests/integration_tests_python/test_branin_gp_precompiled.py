import os
import pickle
import numpy as np
import pytest

from pqueens.main import main


def test_branin_gp_precompiled(inputdir, tmpdir, expected_mean, expected_var):
    """ Test case for GPPrecompiled based GP model """
    arguments = [
        '--input=' + os.path.join(inputdir, 'branin_gp_precompiled.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["mean"], expected_mean, decimal=4
    )
    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["variance"], expected_var, decimal=2
    )


@pytest.fixture()
def expected_mean():
    mean = np.array(
        [
            [39.28118472],
            [62.35579788],
            [37.72489946],
            [51.89237794],
            [52.89154123],
            [43.72648536],
            [47.88182739],
            [48.40690646],
            [47.33544486],
            [36.74236273],
        ]
    )
    return mean


@pytest.fixture()
def expected_var():
    var = np.array(
        [
            17232.68034057,
            11590.8107475,
            11496.18689341,
            12239.15100139,
            11262.84627687,
            15198.2884741,
            13759.70271923,
            13074.23700668,
            11230.7614391,
            11636.01751018,
        ]
    )
    return var
