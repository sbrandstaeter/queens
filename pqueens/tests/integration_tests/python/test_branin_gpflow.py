import os
import pickle
import numpy as np
import pytest

from pqueens.main import main


@pytest.mark.integration_tests
def test_branin_gpflow(inputdir, tmpdir, expected_mean, expected_var):
    """ Test case for GPflow based GP model """
    arguments = [
        '--input=' + os.path.join(inputdir, 'branin_gpflow_surrogate.json'),
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
            [127.97233506],
            [39.73551321],
            [47.00641347],
            [28.88934819],
            [22.40199886],
            [150.69211917],
            [104.25630329],
            [92.22700928],
            [50.69060622],
            [22.18886383]
        ]
    )
    return mean


@pytest.fixture()
def expected_var():
    var = np.array(
        [
            [788.8004288],
            [1.8365012],
            [2.25043994],
            [4.24878946],
            [1.97026586],
            [174.50881662],
            [14.06623098],
            [8.34025715],
            [0.95922611],
            [0.33420735]
        ]
    )
    return var
