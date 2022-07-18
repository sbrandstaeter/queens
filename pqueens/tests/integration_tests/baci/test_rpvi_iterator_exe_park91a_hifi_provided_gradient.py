"""Integration test for reparameerization trick VI as executable."""

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from pqueens import run
from pqueens.utils import injector


@pytest.mark.integration_tests_baci
def test_rpvi_iterator_exe_park91a_hifi_provided_gradient(
    inputdir,
    tmpdir,
    create_experimental_data_park91a_hifi_on_grid,
    third_party_inputs,
    example_simulator_fun_dir,
    gradient_method,
    create_input_file_executable_park91a_hifi_on_grid,
):
    """Test for the rpvi iterator based on the park91a_hifi function."""
    # generate json input file from template
    template = os.path.join(inputdir, "rpvi_exe_park91a_hifi_template.json")
    third_party_input_file = tmpdir.join("input_file_executable_park91a_hifi_on_grid.csv")
    experimental_data_path = tmpdir
    executable = os.path.join(
        example_simulator_fun_dir, "executable_park91a_hifi_on_grid_with_gradients.py"
    )
    plot_dir = tmpdir
    dir_dict = {
        "experimental_data_path": experimental_data_path,
        "plot_dir": plot_dir,
        "input_file": third_party_input_file,
        "executable": executable,
        "experiment_dir": tmpdir,
        "gradient_method": gradient_method,
    }
    input_file = os.path.join(tmpdir, "rpvi_park91a_hifi.json")
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(Path(input_file), Path(tmpdir))

    # This seed is fixed so that the variational distribution is initialized so that the park
    # function can be evaluated correctly
    np.random.seed(211)
    # actual main call of vi_rp

    # get the results of the QUEENS run
    result_file = os.path.join(tmpdir, "inverse_rpvi_park91a_hifi.pickle")
    with open(result_file, "rb") as handle:
        results = pickle.load(handle)

    # Actual tests
    assert np.abs(results["variational_distribution"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.15
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5


@pytest.fixture(params=["finite_difference", "provided_gradient"])
def gradient_method(request):
    """Fixture for parameterized gradient methods."""
    return request.param


@pytest.fixture()
def create_input_file_executable_park91a_hifi_on_grid(tmpdir):
    """Write temporary input file for executable."""
    input_path = tmpdir.join("input_file_executable_park91a_hifi_on_grid.csv")
    with open(input_path, "w", encoding='utf-8') as input_file:
        input_file.write("{x1}\n")
        input_file.write("{x2}")
