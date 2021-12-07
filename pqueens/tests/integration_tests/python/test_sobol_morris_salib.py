import os
import pickle

import numpy as np
import pytest

from pqueens.main import main


# TODO fix these test, because as of now these test produce platform dependent results
@pytest.mark.integration_tests
def test_ishigami_morris_salib(inputdir, tmpdir):
    """Test case for salib based morris iterator."""
    arguments = [
        '--input=' + os.path.join(inputdir, 'sobol_morris_salib.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # print(results)
    expected_result_mu = np.array(
        [
            25.8299150077341,
            19.28297176050532,
            -14.092164789704626,
            5.333475971922498,
            -11.385141403296364,
            13.970208961715421,
            -3.0950202483238303,
            0.6672725255532903,
            7.2385092339309445,
            -7.7664016980947075,
        ]
    )
    expected_result_mu_star = np.array(
        [
            29.84594504725642,
            21.098173537614855,
            16.4727722348437,
            26.266876218598668,
            16.216603266281044,
            18.051629859410895,
            3.488313966697564,
            2.7128638920479147,
            7.671230484535577,
            10.299932289624746,
        ]
    )
    expected_result_sigma = np.array(
        [
            53.88783786787971,
            41.02192670857979,
            29.841807478998156,
            43.33349033575829,
            29.407676882180404,
            31.679653142831512,
            5.241491105224932,
            4.252334015139214,
            10.38274186974731,
            18.83046700807382,
        ]
    )
    np.testing.assert_allclose(results["sensitivity_indices"]['mu'], expected_result_mu)
    np.testing.assert_allclose(results["sensitivity_indices"]['mu_star'], expected_result_mu_star)
    np.testing.assert_allclose(results["sensitivity_indices"]['sigma'], expected_result_sigma)
