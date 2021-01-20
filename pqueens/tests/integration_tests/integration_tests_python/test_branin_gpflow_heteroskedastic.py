import os
import pickle
import numpy as np
import pytest

from pqueens.main import main


def test_branin_gpflow_heteroskedastic(inputdir, tmpdir, expected_mean, expected_var):
    """ Test case for GPflow based heteroskedastic model """
    arguments = [
        '--input=' + os.path.join(inputdir, 'branin_gp_heteroskedastic_surrogate.json'),
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
            [0.20460629],
            [1.20638755],
            [2.11045979],
            [1.85950349],
            [2.4312511],
            [0.31418876],
            [1.0806681],
            [1.22094408],
            [1.18375414],
            [3.01705721],
        ]
    )
    return mean


@pytest.fixture()
def expected_var():
    var = np.array(
        [
            [33365.73939649],
            [6110.0866159],
            [3628.43470359],
            [862.10622222],
            [270.78329203],
            [27494.43093913],
            [14647.44633305],
            [23662.30150605],
            [4752.77857839],
            [392.26766014],
        ]
    )
    return var
