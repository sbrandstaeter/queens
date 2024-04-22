"""TODO_doc."""

import numpy as np
import pytest

from queens.main import run
from queens.utils.input_to_script import create_script_from_input_file
from queens.utils.io_utils import load_result
from queens.utils.run_subprocess import run_subprocess


@pytest.fixture(name="expected_result_mu")
def fixture_expected_result_mu():
    """Mu result fixture."""
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
    return expected_result_mu


@pytest.fixture(name="expected_result_mu_star")
def fixture_expected_result_mu_star():
    """Mu star result fixture."""
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
    return expected_result_mu_star


@pytest.fixture(name="expected_result_sigma")
def fixture_expected_result_sigma():
    """Sigma result fixture."""
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
    return expected_result_sigma


def test_elementary_effects_sobol(
    inputdir, tmp_path, expected_result_mu, expected_result_mu_star, expected_result_sigma
):
    """Test case for elementary effects on Sobol's G-function."""
    run(inputdir / 'elementary_effects_sobol.yml', tmp_path)

    results = load_result(tmp_path / 'xxx.pickle')

    np.testing.assert_allclose(results["sensitivity_indices"]['mu'], expected_result_mu)
    np.testing.assert_allclose(results["sensitivity_indices"]['mu_star'], expected_result_mu_star)
    np.testing.assert_allclose(results["sensitivity_indices"]['sigma'], expected_result_sigma)


def test_elementary_effects_sobol_from_script(
    inputdir, tmp_path, expected_result_mu, expected_result_mu_star, expected_result_sigma
):
    """Test case for elementary effects using a script."""
    input_file = inputdir / 'elementary_effects_sobol.yml'
    script_path = tmp_path / "script.py"
    create_script_from_input_file(input_file, tmp_path, script_path)

    # Command to call the script
    command = f"python {str(script_path.resolve())}"

    # The False is needed due to the loading bars
    run_subprocess(command, raise_error_on_subprocess_failure=False)

    results = load_result(tmp_path / 'xxx.pickle')

    np.testing.assert_allclose(results["sensitivity_indices"]['mu'], expected_result_mu)
    np.testing.assert_allclose(results["sensitivity_indices"]['mu_star'], expected_result_mu_star)
    np.testing.assert_allclose(results["sensitivity_indices"]['sigma'], expected_result_sigma)
