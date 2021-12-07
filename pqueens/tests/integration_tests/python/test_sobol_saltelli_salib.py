import os
import pickle

import numpy as np
import pytest

from pqueens.main import main


@pytest.mark.integration_tests
def test_sobol_saltelli_salib_2nd_order(inputdir, tmpdir):
    """Test Saltelli SALib iterator with Sobol G function.

    Including first, second and total order indices. The test should
    converge to the analytical solution defined in the Sobol G function
    implementaion (see sobol.py).
    """
    arguments = [
        '--input=' + os.path.join(inputdir, 'sobol_saltelli_salib.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    expected_result = dict()

    expected_result["S1"] = np.array(
        [
            0.03417320602166007,
            0.12182632006554496,
            0.03985974185502539,
            0.010365005479182598,
            0.038394205845418534,
            0.026889576071081243,
            -0.044512526357984604,
            -0.0010940870962751093,
            0.004719184843312276,
            0.004873427040291372,
        ]
    )

    expected_result["S1_conf"] = np.array(
        [
            0.12426324912912282,
            0.2662268575091613,
            0.11029758885586062,
            0.06592860167976117,
            0.07002043191800363,
            0.02358445785632366,
            0.06380826747340128,
            0.01610289912590691,
            0.02160140141643862,
            0.01323125575661702,
        ]
    )
    expected_result["ST"] = np.array(
        [
            0.8706049386700341,
            0.9084884287246713,
            0.23481964978527603,
            0.3220370492267818,
            0.28575542382610636,
            0.5073652847743768,
            0.9794488413638497,
            0.01794530493422253,
            0.009673198188887209,
            0.010160604709604676,
        ]
    )

    expected_result["ST_conf"] = np.array(
        [
            0.7553506677923951,
            4.157520166210192,
            0.20572184906089427,
            0.2017302745467454,
            0.5049730031271069,
            0.19954486390417922,
            0.8117165298764415,
            0.04441454378912033,
            0.04344152177178538,
            0.02706861507841958,
        ]
    )

    np.testing.assert_allclose(results["sensitivity_indices"]["S1"], expected_result["S1"])
    np.testing.assert_allclose(
        results["sensitivity_indices"]["S1_conf"], expected_result["S1_conf"]
    )
    np.testing.assert_allclose(results["sensitivity_indices"]["ST"], expected_result["ST"])
    np.testing.assert_allclose(
        results["sensitivity_indices"]["ST_conf"], expected_result["ST_conf"]
    )
