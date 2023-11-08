"""Integration test for reparameterization trick VI."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mock import patch

from queens.distributions.normal import NormalDistribution
from queens.main import run
from queens.models.likelihood_models.gaussian_likelihood import GaussianLikelihood
from queens.utils import injector


def test_rpvi_iterator_park91a_hifi(
    inputdir,
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    module_path,
):
    """Integration test for the rpvi iterator.

    Based on the *park91a_hifi* function.
    """
    template = Path(inputdir, "rpvi_park91a_hifi_template.yml")
    experimental_data_path = tmp_path
    plot_dir = tmp_path
    dir_dict = {
        "experimental_data_path": experimental_data_path,
        "plot_dir": plot_dir,
        "forward_model_name": "fd_model",
        "my_function": "park91a_hifi_on_grid",
        "model": "model",
        "external_python_module": module_path,
    }
    input_file = tmp_path / "rpvi_park91a_hifi.yml"
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(input_file, tmp_path)

    # This seed is fixed so that the variational distribution is initialized so that the park
    # function can be evaluated correctly
    np.random.seed(211)
    run(input_file, tmp_path)
    # actual main call

    # get the results of the QUEENS run
    result_file = tmp_path / "inverse_rpvi_park91a_hifi.pickle"
    with open(result_file, "rb") as handle:
        results = pickle.load(handle)

    # Actual tests
    assert np.abs(results["variational_distribution"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.1
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5


def test_rpvi_iterator_park91a_hifi_external_module(
    inputdir,
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    _write_custom_likelihood_model,
    module_path,
):
    """Integration test for the rpvi iterator.

    Based on the *park91a_hifi* function.
    """
    template = inputdir / "rpvi_park91a_hifi_template.yml"
    experimental_data_path = tmp_path
    plot_dir = tmp_path
    dir_dict = {
        "experimental_data_path": experimental_data_path,
        "plot_dir": plot_dir,
        "forward_model_name": "fd_model",
        "my_function": "park91a_hifi_on_grid",
        "model": "model_external",
        "external_python_module": module_path,
    }
    input_file = tmp_path / "rpvi_park91a_hifi.yml"
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(input_file, tmp_path)

    # This seed is fixed so that the variational distribution is initialized so that the park
    # function can be evaluated correctly
    np.random.seed(211)
    run(input_file, tmp_path)
    # actual main call

    # get the results of the QUEENS run
    result_file = tmp_path / "inverse_rpvi_park91a_hifi.pickle"
    with open(result_file, "rb") as handle:
        results = pickle.load(handle)

    # Actual tests
    assert np.abs(results["variational_distribution"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.1
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5


def test_rpvi_iterator_park91a_hifi_provided_gradient(
    inputdir, tmp_path, _create_experimental_data_park91a_hifi_on_grid, module_path
):
    """Test for the rpvi iterator based on the *park91a_hifi* function."""
    # generate json input file from template
    template = inputdir / "rpvi_park91a_hifi_template.yml"
    experimental_data_path = tmp_path
    plot_dir = tmp_path
    dir_dict = {
        "experimental_data_path": experimental_data_path,
        "plot_dir": plot_dir,
        "forward_model_name": "simulation_model",
        "my_function": "park91a_hifi_on_grid_with_gradients",
        "model": "model",
        "external_python_module": module_path,
    }
    input_file = tmp_path / "rpvi_park91a_hifi.yml"
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(input_file, tmp_path)

    # This seed is fixed so that the variational distribution is initialized so that the park
    # function can be evaluated correctly
    np.random.seed(211)
    # actual main call of vi_rp

    # get the results of the QUEENS run
    result_file = tmp_path / "inverse_rpvi_park91a_hifi.pickle"
    with open(result_file, "rb") as handle:
        results = pickle.load(handle)

    # Actual tests
    assert np.abs(results["variational_distribution"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.1
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5


likelihood_mean = np.array([-2.0, 1.0])
likelihood_covariance = np.diag(np.array([0.1, 10.0]))
likelihood = NormalDistribution(likelihood_mean, likelihood_covariance)


def target_density(
    self, samples
):  # pylint: disable=unused-argument  # pylint: disable=unused-argument
    """Target posterior density."""
    log_likelihood_output = likelihood.logpdf(samples)
    grad_log_likelihood = likelihood.grad_logpdf(samples)

    return log_likelihood_output, grad_log_likelihood


@pytest.fixture(name="forward_model", scope="module", params=['simulation_model', 'fd_model'])
def fixture_forward_model(request):
    """Gradient method."""
    return request.param


def test_gaussian_rpvi(inputdir, tmp_path, _create_experimental_data, forward_model):
    """Test RPVI with univariate Gaussian."""
    template = inputdir / "rpvi_gaussian_template.yml"

    dir_dict = {
        "plot_dir": tmp_path,
        "experimental_data_path": tmp_path,
        "forward_model_name": forward_model,
    }
    input_file = tmp_path / "rpvi_gaussian.yml"
    injector.inject(dir_dict, template, input_file)

    # mock methods related to likelihood
    with patch.object(GaussianLikelihood, "evaluate_and_gradient", target_density):
        run(input_file, tmp_path)

    # get the results of the QUEENS run
    result_file = tmp_path / "rpvi_gaussian.pickle"
    with open(result_file, "rb") as handle:
        results = pickle.load(handle)

    posterior_covariance = np.diag(np.array([1 / 11, 100 / 11]))
    posterior_mean = np.array([-20 / 11, 20 / 11]).reshape(-1, 1)

    # Actual tests
    np.testing.assert_almost_equal(
        results["variational_distribution"]["mean"], posterior_mean, decimal=3
    )
    np.testing.assert_almost_equal(
        results["variational_distribution"]["covariance"], posterior_covariance, decimal=4
    )


@pytest.fixture(name="_create_experimental_data")
def fixture_create_experimental_data(tmp_path):
    """Fixture for dummy data."""
    data_dict = {'y_obs': np.zeros(1)}
    experimental_data_path = tmp_path / 'experimental_data.csv'
    dataframe = pd.DataFrame.from_dict(data_dict)
    dataframe.to_csv(experimental_data_path, index=False)


@pytest.fixture(name="module_path")
def fixture_module_path(tmp_path):
    """Generate path for new likelihood module."""
    my_module_path = tmp_path / "my_likelihood_module.py"
    return str(my_module_path)


@pytest.fixture(name="_write_custom_likelihood_model")
def fixture_write_custom_likelihood_model(module_path):
    """Write custom likelihood class to file."""
    custom_class_lst = [
        "from queens.models.likelihood_models.gaussian_likelihood import GaussianLikelihood\n",
        "class MyLikelihood(GaussianLikelihood):\n",
        "   pass",
    ]
    with open(module_path, 'w', encoding='utf-8') as f:
        for my_string in custom_class_lst:
            f.writelines(my_string)
