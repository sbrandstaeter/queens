import os
import pickle

import numpy as np
import pytest

from pqueens.main import main


@pytest.mark.integration_tests
def test_sobol_indices_sobol(inputdir, tmpdir):
    """
    Test Sobol Index iterator with Sobol G function.

    Including first, second and total order indices.
    The test should converge to the analytical solution defined in the Sobol G function
    implementation (see sobol.py).
    """
    arguments = [
        '--input=' + os.path.join(inputdir, 'sobol_indices_sobol.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    expected_result = dict()

    expected_result["S1"] = np.array(
        [
            0.03287880027356417,
            0.12275796466238141,
            0.03885626043646576,
            0.009744086663312954,
            0.03779163384481807,
            0.025828849557999536,
            -0.04139967646816724,
            -0.0008383917148741094,
            0.0046093764686569545,
            0.004890095900063113,
        ]
    )
    expected_result["S1_conf"] = np.array(
        [
            0.0267367056,
            0.1614473756,
            0.0446664140,
            0.1030576095,
            0.0715392455,
            0.0743431043,
            0.0757356440,
            0.0176071872,
            0.0236584394,
            0.0051844290,
        ]
    )

    expected_result["ST"] = np.array(
        [
            0.8706049386700344,
            0.9084884287246714,
            0.234819649785276,
            0.32203704922678184,
            0.28575542382610625,
            0.5073652847743768,
            0.9794488413638496,
            0.01794530493422254,
            0.009673198188887198,
            0.010160604709604686,
        ]
    )

    expected_result["ST_conf"] = np.array(
        [
            0.4359593595,
            3.4830312608,
            0.1600557592,
            0.2731329567,
            0.3505338739,
            0.2773761454,
            0.7057099402,
            0.0190672368,
            0.0295136874,
            0.0277708321,
        ]
    )

    expected_result["S2"] = np.array(
        [
            [
                np.nan,
                0.21271334153627885,
                0.010722528700970534,
                0.0033859688589870634,
                0.09426247609368452,
                0.012373958761854958,
                0.08112891427803003,
                0.02061955585077749,
                0.013502783476358828,
                0.015425852124696887,
            ],
            [
                np.nan,
                np.nan,
                -0.07857784761587458,
                -0.030204143434144425,
                -0.09500091656783104,
                -0.08406391188750016,
                -0.055057701245750226,
                -0.08045445823360135,
                -0.10083440537279068,
                -0.09518754829682612,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                -0.033083554023666605,
                -0.010695143705473288,
                -0.0033310891827415604,
                -0.020667400589763407,
                -0.01549194577878769,
                -0.021679794648766012,
                -0.021236951250418676,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.04311995205988921,
                0.03771789390510989,
                0.00532453705473699,
                0.013352178049062505,
                0.012880977687501235,
                0.012779928856921833,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                -0.010919694144387655,
                -0.025786686734173683,
                -0.01992873429946779,
                -0.02825113257884349,
                -0.026756740633499816,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                -0.01610767850675026,
                -0.03301527521007352,
                -0.033088333558550224,
                -0.030857585321899847,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.046920923371739975,
                0.047717234944161954,
                0.04655742588404423,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.004629179339114694,
                0.0022919874320803313,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                -0.0037027010271028964,
            ],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    )

    expected_result["S2_conf"] = np.array(
        [
            [
                np.nan,
                0.2302033993,
                0.1767744071,
                0.0727004660,
                0.5715173997,
                0.1981648654,
                0.2292276020,
                0.1291920907,
                0.1917071250,
                0.1787768697,
            ],
            [
                np.nan,
                np.nan,
                0.3769790878,
                0.8082964090,
                0.3153796763,
                0.3830562431,
                0.6861041390,
                0.4140223269,
                0.4013110877,
                0.4321964362,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                0.0521460071,
                0.0934262116,
                0.0675665076,
                0.0503032832,
                0.0460493846,
                0.0394698900,
                0.0493946655,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.4261374303,
                0.1481089941,
                0.1472484464,
                0.1434691665,
                0.2108224367,
                0.1877406256,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.1296112773,
                0.0901383559,
                0.1073482468,
                0.1060688378,
                0.1006672459,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.0971184412,
                0.0777429467,
                0.0745773590,
                0.0809483392,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.0315764492,
                0.0334681878,
                0.0338116736,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.0273391091,
                0.0248472949,
            ],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0482628341],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
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

    np.testing.assert_allclose(results["sensitivity_indices"]["S2"], expected_result["S2"])
    np.testing.assert_allclose(
        results["sensitivity_indices"]["S2_conf"], expected_result["S2_conf"]
    )
