import os
import pickle

import numpy as np
import pytest

from pqueens.main import main


@pytest.mark.integration_tests
def test_sobol_saltelli_2nd_order(inputdir, tmpdir):
    """
    Test saltelli iterator with Sobol G function.

    Including first, second and total order indices.
    The test should converge to the analytical solution defined in the Sobol G function
    implementaion (see sobol.py).
    """
    arguments = [
        '--input=' + os.path.join(inputdir, 'sobol_saltelli_2nd_order.json'),
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
            0.10986792487961591,
            0.25000599956728076,
            0.11024277529867381,
            0.06149594923578095,
            0.06918635500808559,
            0.021478128416154076,
            0.06354361408306426,
            0.015569627261478092,
            0.02128151875504991,
            0.013440584808635126,
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
            0.755350667792397,
            4.157520166210199,
            0.20572184906089488,
            0.20173027454674597,
            0.5049730031271076,
            0.19954486390417944,
            0.8117165298764414,
            0.04441454378912035,
            0.043441521771785424,
            0.027068615078419605,
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
                0.1967466913123026,
                0.09565629386983263,
                0.06000822931018138,
                0.6837761105291199,
                0.2475368991335622,
                0.2547257094941949,
                0.11106811220019396,
                0.11807055333364978,
                0.10155559221245174,
            ],
            [
                np.nan,
                np.nan,
                0.3985842438995974,
                1.2580664159711359,
                0.6259599962105775,
                0.32053683219916534,
                0.5692214310081142,
                0.5572583283528845,
                0.35124861465874685,
                0.4001331058829118,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                0.10173701407184092,
                0.2646590272333874,
                0.15756584781356767,
                0.07611662684699781,
                0.08400677379232034,
                0.06960740860683465,
                0.08386421598287554,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.28222395700947905,
                0.26897274076034905,
                0.09855318755961003,
                0.1488129387029852,
                0.1125570457017122,
                0.11781893215648177,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.12085394747459759,
                0.06926255643877466,
                0.08249544718780262,
                0.06400291986713515,
                0.06201060403343306,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.05601453092006938,
                0.13270846731234276,
                0.10230366474274967,
                0.09697215881808433,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.0835121759225804,
                0.07726586432600173,
                0.08457703041833556,
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
                0.03232346368831656,
                0.038049159818777614,
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
                0.0489285763078056,
            ],
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
