"""Integration test for reparametrization trick VI as executable."""

import numpy as np
import pytest

from queens.main import run
from queens.utils import injector
from queens.utils.io_utils import load_result
from queens.utils.run_subprocess import run_subprocess


@pytest.fixture(name="python_path")
def fixture_python_path():
    """Current python path."""
    _, _, stdout, _ = run_subprocess("which python")
    return stdout.strip()


def test_rpvi_iterator_exe_park91a_hifi_provided_gradient(
    inputdir,
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    example_simulator_fun_dir,
    _create_input_file_executable_park91a_hifi_on_grid,
    python_path,
):
    """Test for the *rpvi* iterator based on the *park91a_hifi* function."""
    # pylint: disable=duplicate-code
    # generate json input file from template
    template = inputdir / "rpvi_exe_park91a_hifi_template.yml"
    third_party_input_file = tmp_path / "input_file_executable_park91a_hifi_on_grid.csv"
    experimental_data_path = tmp_path
    executable = example_simulator_fun_dir / "executable_park91a_hifi_on_grid_with_gradients.py"
    executable = f"{python_path} {executable} p"
    plot_dir = tmp_path
    dir_dict = {
        "experimental_data_path": experimental_data_path,
        "plot_dir": plot_dir,
        "input_file": third_party_input_file,
        "executable": executable,
        "adjoint_executable": "_",
        "forward_model_name": "simulation_model",
        "driver": "driver_with_gradient",
    }
    input_file = tmp_path / "rpvi_park91a_hifi.yml"
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(input_file, tmp_path)

    # This seed is fixed so that the variational distribution is initialized so that the park
    # function can be evaluated correctly
    np.random.seed(211)

    # get the results of the QUEENS run
    results = load_result(tmp_path / "inverse_rpvi_park91a_hifi.pickle")

    # Actual tests
    assert np.abs(results["variational_distribution"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.15
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5


@pytest.mark.max_time_for_test(20)
def test_rpvi_iterator_exe_park91a_hifi_finite_differences_gradient(
    inputdir,
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    example_simulator_fun_dir,
    _create_input_file_executable_park91a_hifi_on_grid,
    python_path,
):
    """Test for the *rpvi* iterator based on the *park91a_hifi* function."""
    # pylint: disable=duplicate-code
    # generate json input file from template
    template = inputdir / "rpvi_exe_park91a_hifi_template.yml"
    third_party_input_file = tmp_path / "input_file_executable_park91a_hifi_on_grid.csv"
    experimental_data_path = tmp_path
    executable = example_simulator_fun_dir / "executable_park91a_hifi_on_grid_with_gradients.py"
    executable = f"{python_path} {executable} s"
    plot_dir = tmp_path
    dir_dict = {
        "experimental_data_path": experimental_data_path,
        "plot_dir": plot_dir,
        "input_file": third_party_input_file,
        "adjoint_executable": "_",
        "executable": executable,
        "forward_model_name": "fd_model",
        "driver": "driver",
    }
    input_file = tmp_path / "rpvi_park91a_hifi.yml"
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(input_file, tmp_path)

    # This seed is fixed so that the variational distribution is initialized so that the park
    # function can be evaluated correctly
    np.random.seed(211)

    # get the results of the QUEENS run
    results = load_result(tmp_path / "inverse_rpvi_park91a_hifi.pickle")

    # Actual tests
    assert np.abs(results["variational_distribution"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.15
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5


def test_rpvi_iterator_exe_park91a_hifi_adjoint_gradient(
    inputdir,
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    example_simulator_fun_dir,
    _create_input_file_executable_park91a_hifi_on_grid,
    python_path,
):
    """Test the *rpvi* iterator based on the *park91a_hifi* function."""
    # pylint: disable=duplicate-code
    # generate json input file from template
    template = inputdir / "rpvi_exe_park91a_hifi_template.yml"
    third_party_input_file = tmp_path / "input_file_executable_park91a_hifi_on_grid.csv"
    experimental_data_path = tmp_path
    # standard executable of forward run
    executable = example_simulator_fun_dir / "executable_park91a_hifi_on_grid_with_gradients.py"
    executable = f"{python_path} {executable} s"
    # adjoint executable (here we actually use the same executable but call it with
    # a different flag "a" for adjoint)
    adjoint_executable = (
        example_simulator_fun_dir / "executable_park91a_hifi_on_grid_with_gradients.py"
    )
    adjoint_executable = f"{python_path} {adjoint_executable} a"
    plot_dir = tmp_path
    dir_dict = {
        "experimental_data_path": experimental_data_path,
        "plot_dir": plot_dir,
        "input_file": third_party_input_file,
        "executable": executable,
        "forward_model_name": "adjoint_model",
        "adjoint_executable": adjoint_executable,
        "driver": "driver",
    }
    input_file = tmp_path / "rpvi_park91a_hifi.yml"
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(input_file, tmp_path)

    # This seed is fixed so that the variational distribution is initialized so that the park
    # function can be evaluated correctly
    np.random.seed(211)

    # get the results of the QUEENS run
    results = load_result(tmp_path / "inverse_rpvi_park91a_hifi.pickle")

    # Actual tests
    assert np.abs(results["variational_distribution"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.15
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5


@pytest.fixture(name="_create_input_file_executable_park91a_hifi_on_grid")
def fixture_create_input_file_executable_park91a_hifi_on_grid(tmp_path):
    """Write temporary input file for executable."""
    input_path = tmp_path / "input_file_executable_park91a_hifi_on_grid.csv"
    input_path.write_text("{{ x1 }}\n{{ x2 }}", encoding="utf-8")
