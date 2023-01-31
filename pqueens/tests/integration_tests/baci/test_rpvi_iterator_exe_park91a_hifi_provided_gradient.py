"""Integration test for reparametrization trick VI as executable."""

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from pqueens import run
from pqueens.utils import injector


def test_rpvi_iterator_exe_park91a_hifi_callable_gradient(
    inputdir,
    tmpdir,
    create_experimental_data_park91a_hifi_on_grid,
    third_party_inputs,
    example_simulator_fun_dir,
    create_input_file_executable_park91a_hifi_on_grid,
):
    """Test for the *rpvi* iterator based on the *park91a_hifi* function."""
    # generate json input file from template
    template = os.path.join(inputdir, "rpvi_exe_park91a_hifi_template.yml")
    third_party_input_file = tmpdir.join("input_file_executable_park91a_hifi_on_grid.csv")
    experimental_data_path = tmpdir
    executable = os.path.join(
        example_simulator_fun_dir, "executable_park91a_hifi_on_grid_with_gradients.py c"
    )
    plot_dir = tmpdir
    dir_dict = {
        "experimental_data_path": experimental_data_path,
        "plot_dir": plot_dir,
        "input_file": third_party_input_file,
        "executable": executable,
        "experiment_dir": tmpdir,
        "forward_model_name": "forward_model",
        "gradient_data_processor": "gradient_data_processor",
        "gradient_interface": "",
    }
    input_file = os.path.join(tmpdir, "rpvi_park91a_hifi.yml")
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(Path(input_file), Path(tmpdir))

    # This seed is fixed so that the variational distribution is initialized so that the park
    # function can be evaluated correctly
    np.random.seed(211)

    # get the results of the QUEENS run
    result_file = os.path.join(tmpdir, "inverse_rpvi_park91a_hifi.pickle")
    with open(result_file, "rb") as handle:
        results = pickle.load(handle)

    # Actual tests
    assert np.abs(results["variational_distribution"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.15
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5


def test_rpvi_iterator_exe_park91a_hifi_finite_differences_gradient(
    inputdir,
    tmpdir,
    create_experimental_data_park91a_hifi_on_grid,
    third_party_inputs,
    example_simulator_fun_dir,
    create_input_file_executable_park91a_hifi_on_grid,
):
    """Test for the *rpvi* iterator based on the *park91a_hifi* function."""
    # generate json input file from template
    template = os.path.join(inputdir, "rpvi_exe_park91a_hifi_template.yml")
    third_party_input_file = tmpdir.join("input_file_executable_park91a_hifi_on_grid.csv")
    experimental_data_path = tmpdir
    executable = os.path.join(
        example_simulator_fun_dir, "executable_park91a_hifi_on_grid_with_gradients.py s"
    )
    plot_dir = tmpdir
    dir_dict = {
        "experimental_data_path": experimental_data_path,
        "plot_dir": plot_dir,
        "input_file": third_party_input_file,
        "executable": executable,
        "experiment_dir": tmpdir,
        "forward_model_name": "gradient_model",
        "gradient_model_type": "finite_differences",
        "gradient_data_processor": "gradient_data_processor",
        "gradient_interface": "",
    }
    input_file = os.path.join(tmpdir, "rpvi_park91a_hifi.yml")
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(Path(input_file), Path(tmpdir))

    # This seed is fixed so that the variational distribution is initialized so that the park
    # function can be evaluated correctly
    np.random.seed(211)

    # get the results of the QUEENS run
    result_file = os.path.join(tmpdir, "inverse_rpvi_park91a_hifi.pickle")
    with open(result_file, "rb") as handle:
        results = pickle.load(handle)

    # Actual tests
    assert np.abs(results["variational_distribution"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.15
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5


def test_rpvi_iterator_exe_park91a_hifi_adjoint_gradient(
    inputdir,
    tmpdir,
    create_experimental_data_park91a_hifi_on_grid,
    third_party_inputs,
    example_simulator_fun_dir,
    create_input_file_executable_park91a_hifi_on_grid,
):
    """Test for the *rpvi* iterator based on the *park91a_hifi* function."""
    # generate json input file from template
    template = os.path.join(inputdir, "rpvi_exe_park91a_hifi_template.yml")
    third_party_input_file = tmpdir.join("input_file_executable_park91a_hifi_on_grid.csv")
    experimental_data_path = tmpdir
    # standard executable of forward run
    executable = os.path.join(
        example_simulator_fun_dir, "executable_park91a_hifi_on_grid_with_gradients.py s"
    )
    # adjoint executable (here we actually use the same executable but call it with
    # a different flag "a" for adjoint)
    adjoint_executable = os.path.join(
        example_simulator_fun_dir, "executable_park91a_hifi_on_grid_with_gradients.py a"
    )
    plot_dir = tmpdir
    dir_dict = {
        "experimental_data_path": experimental_data_path,
        "plot_dir": plot_dir,
        "input_file": third_party_input_file,
        "executable": executable,
        "experiment_dir": tmpdir,
        "forward_model_name": "gradient_model",
        "gradient_model_type": "adjoint",
        "adjoint_executable": adjoint_executable,
        "gradient_interface": "gradient_interface",
        "gradient_data_processor": "",
    }
    input_file = os.path.join(tmpdir, "rpvi_park91a_hifi.yml")
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(Path(input_file), Path(tmpdir))

    # This seed is fixed so that the variational distribution is initialized so that the park
    # function can be evaluated correctly
    np.random.seed(211)

    # get the results of the QUEENS run
    result_file = os.path.join(tmpdir, "inverse_rpvi_park91a_hifi.pickle")
    with open(result_file, "rb") as handle:
        results = pickle.load(handle)

    # Actual tests
    assert np.abs(results["variational_distribution"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.15
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5


@pytest.fixture()
def create_input_file_executable_park91a_hifi_on_grid(tmpdir):
    """Write temporary input file for executable."""
    input_path = tmpdir.join("input_file_executable_park91a_hifi_on_grid.csv")
    with open(input_path, "w", encoding='utf-8') as input_file:
        input_file.write("{{ x1 }}\n")
        input_file.write("{{ x2 }}")
