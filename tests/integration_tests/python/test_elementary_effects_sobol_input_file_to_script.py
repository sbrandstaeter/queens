"""Integration test for the elementary effects iterator.

This test is based on Sobol's G function and uses an input file.
"""

import numpy as np

from queens.utils.input_to_script import create_script_from_input_file
from queens.utils.io_utils import load_result
from queens.utils.run_subprocess import run_subprocess


def test_script_from_input_elementary_effects_sobol(
    inputdir, tmp_path, expected_result_mu, expected_result_mu_star, expected_result_sigma
):
    """Test elementary effects using a script created from input file."""
    input_file = inputdir / "elementary_effects_sobol.yml"
    script_path = tmp_path / "script.py"
    create_script_from_input_file(input_file, tmp_path, script_path)

    # Command to call the script
    command = f"python {str(script_path.resolve())}"

    # The False is needed due to the loading bars
    run_subprocess(command, raise_error_on_subprocess_failure=False)

    results = load_result(tmp_path / "xxx.pickle")

    np.testing.assert_allclose(results["sensitivity_indices"]["mu"], expected_result_mu)
    np.testing.assert_allclose(results["sensitivity_indices"]["mu_star"], expected_result_mu_star)
    np.testing.assert_allclose(results["sensitivity_indices"]["sigma"], expected_result_sigma)
