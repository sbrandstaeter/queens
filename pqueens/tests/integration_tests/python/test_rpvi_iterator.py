"""Integration test for reparameterization trick VI."""

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from pqueens import run
from pqueens.utils import injector


@pytest.mark.integration_tests
def test_rpvi_iterator_park91a_hifi(
    inputdir,
    tmpdir,
    create_experimental_data_park91a_hifi_on_grid,
):
    """Integration test for the rpvi iterator.

    Based on the park91a_hifi function.
    """
    template = os.path.join(inputdir, "rpvi_park91a_hifi_template.json")
    experimental_data_path = tmpdir
    plot_dir = tmpdir
    dir_dict = {
        "experimental_data_path": experimental_data_path,
        "plot_dir": plot_dir,
        "gradient_method": "finite_difference",
        "my_function": "park91a_hifi_on_grid",
        "likelihood_model_type": "gaussian",
        "external_python_module": "",
    }
    input_file = os.path.join(tmpdir, "rpvi_park91a_hifi.json")
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(Path(input_file), Path(tmpdir))

    # This seed is fixed so that the variational distribution is initialized so that the park
    # function can be evaluated correctly
    np.random.seed(211)
    run(Path(input_file), Path(tmpdir))
    # actual main call

    # get the results of the QUEENS run
    result_file = os.path.join(tmpdir, "inverse_rpvi_park91a_hifi.pickle")
    with open(result_file, "rb") as handle:
        results = pickle.load(handle)

    # Actual tests
    assert np.abs(results["variational_distribution"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.1
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5


@pytest.mark.integration_tests
def test_rpvi_iterator_park91a_hifi_external_module(
    inputdir,
    tmpdir,
    create_experimental_data_park91a_hifi_on_grid,
    write_custom_likelihood_model,
    module_path,
):
    """Integration test for the rpvi iterator.

    Based on the park91a_hifi function.
    """
    template = os.path.join(inputdir, "rpvi_park91a_hifi_template.json")
    experimental_data_path = tmpdir
    plot_dir = tmpdir
    dir_dict = {
        "experimental_data_path": experimental_data_path,
        "plot_dir": plot_dir,
        "gradient_method": "finite_difference",
        "my_function": "park91a_hifi_on_grid",
        "likelihood_model_type": "MyLikelihood",
        "external_python_module": module_path,
    }
    input_file = os.path.join(tmpdir, "rpvi_park91a_hifi.json")
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(Path(input_file), Path(tmpdir))

    # This seed is fixed so that the variational distribution is initialized so that the park
    # function can be evaluated correctly
    np.random.seed(211)
    run(Path(input_file), Path(tmpdir))
    # actual main call

    # get the results of the QUEENS run
    result_file = os.path.join(tmpdir, "inverse_rpvi_park91a_hifi.pickle")
    with open(result_file, "rb") as handle:
        results = pickle.load(handle)

    # Actual tests
    assert np.abs(results["variational_distribution"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.1
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5


@pytest.mark.integration_tests
def test_rpvi_iterator_park91a_hifi_provided_gradient(
    inputdir, tmpdir, create_experimental_data_park91a_hifi_on_grid
):
    """Test for the rpvi iterator based on the park91a_hifi function."""
    # generate json input file from template
    template = os.path.join(inputdir, "rpvi_park91a_hifi_template.json")
    experimental_data_path = tmpdir
    plot_dir = tmpdir
    dir_dict = {
        "experimental_data_path": experimental_data_path,
        "plot_dir": plot_dir,
        "gradient_method": "provided_gradient",
        "my_function": "park91a_hifi_on_grid_with_gradients",
        "likelihood_model_type": "gaussian",
        "external_python_module": "",
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
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.1
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5


@pytest.fixture()
def module_path(tmpdir):
    """Generate path for new likelihood module."""
    my_module_path = Path(tmpdir, "my_likelihood_module.py")
    return str(my_module_path)


@pytest.fixture()
def write_custom_likelihood_model(module_path):
    """Write custom likelihood class to file."""
    # pylint: disable=line-too-long
    custom_class_lst = [
        "from pqueens.models.likelihood_models.gaussian_likelihood import GaussianLikelihood\n",
        "class MyLikelihood(GaussianLikelihood):\n",
        "   pass",
    ]
    # pylint: enable=line-too-long
    with open(module_path, 'w') as f:
        for my_string in custom_class_lst:
            f.writelines(my_string)
