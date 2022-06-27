import os
import pickle

import numpy as np
import pandas as pd
import pytest
from mock import patch
from scipy.stats import multivariate_normal as mvn

import pqueens.visualization.variational_inference_visualization as vis
from pqueens.main import main
from pqueens.tests.integration_tests.example_simulator_functions.park91a import park91a_hifi_on_grid
from pqueens.utils import injector, variational_inference_utils


# TODO add the test prefix again after fullrank was implemented
@pytest.mark.integration_tests
def rpvi_density_match(
    mocker,
    inputdir,
    tmpdir,
    dummy_rpvi_instance,
    visualization_obj,
):
    # fix the random seed
    np.random.seed(1)

    # mock all parts of the algorithm that has to do with initialization or an underlying model
    mocker.patch(
        "pqueens.iterators.variational_inference_reparameterization.rpviIterator.initialize_run",
        return_value=None,
    )

    # actual main call of bbvi with patched density for posterior
    with patch.object(RPVIIterator, 'get_log_posterior_unnormalized', target_density):
        variational_distr_obj = dummy_rpvi_instance.variational_distribution_obj
        mu = np.array([0.1, 0.7, 0.2, 0.3, 0.25])
        cov = np.exp(np.diag([0.5, 0.5, 0.5, 0.5, 0.5]) * 2)
        var_params = variational_distr_obj.construct_variational_params(mu, cov)
        dummy_rpvi_instance.variational_params = var_params
        dummy_rpvi_instance.variational_params_array = np.empty((len(var_params), 0))
        # actual run of the algorithm
        dummy_rpvi_instance.run()

        opt_variational_params = np.array(dummy_rpvi_instance.variational_params)

        elbo = dummy_rpvi_instance.elbo_list

    # Actual tests
    opt_variational_samples = variational_distr_obj.draw(opt_variational_params, 10000)
    variational_logpdf = variational_distr_obj.logpdf(
        opt_variational_params, opt_variational_samples
    )
    target_logpdf = target_density("dummy", x=opt_variational_samples, pdf=False).flatten()
    kl_divergence = np.abs(np.mean(variational_logpdf - target_logpdf))
    # Test if KL divergence is not too large
    assert kl_divergence < 10**5
    # Test if the elbo declined
    assert elbo[0] < elbo[-1]


@pytest.mark.integration_tests
def test_rpvi_iterator_park91a_hifi(inputdir, tmpdir, design_and_write_experimental_data_to_csv):
    """Integration test for the rpvi iterator based on the park91a_hifi
    function."""

    # generate json input file from template
    template = os.path.join(inputdir, "rpvi_park91a_hifi_template.json")
    experimental_data_path = tmpdir
    plot_dir = tmpdir
    dir_dict = {
        "experimental_data_path": experimental_data_path,
        "plot_dir": plot_dir,
    }
    input_file = os.path.join(tmpdir, "rpvi_park91a_hifi.json")
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    arguments = [
        "--input=" + input_file,
        "--output=" + str(tmpdir),
    ]

    # This seed is fixed so that the variational distribution is initalized so that the park
    # function can be evaluted correctly
    np.random.seed(211)
    # actual main call of bbvi
    main(arguments)

    # get the results of the QUEENS run
    result_file = os.path.join(tmpdir, "inverse_rpvi_park91a_hifi.pickle")
    with open(result_file, "rb") as handle:
        results = pickle.load(handle)
    elbo_list = results["iteration_data"]["elbo"]

    # Actual tests
    assert np.abs(results["variational_distribution"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.1
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5


@pytest.fixture()
def dummy_rpvi_instance(tmpdir, my_variational_distribution_obj):
    # TODO this needs to be adjusted for rpvi !!
    #  ----- interesting params one might want to change ---------------------------
    n_samples_per_iter = 5
    # -1 indicates to run a fixed number of samples
    min_requ_relative_change_variational_params = -1
    max_feval = 10 * n_samples_per_iter
    num_variables = 5
    learning_rate = 0.01
    geometry_obj = None
    variational_approximation_type = 'mean_field'
    iteration_num = 20
    variational_samples = None
    variational_params_lst = []
    variational_distribution_obj = None
    relative_change_variational_params_lst = []
    log_likelihood_vec = None
    elbo_lst = []
    log_posterior_unnormalized = None
    v_param_adams = None
    m_param_adams = None
    prior_obj_list = []

    # ------ params we want to keep fixed -----------------------------------------
    variational_transformation = None
    variational_family = 'normal'
    experiment_name = 'density_match'
    result_description = {
        "write_results": False,
        "plotting_options": {
            "plot_boolean": False,
            "plotting_dir": tmpdir,
            "plot_name": "variational_params_convergence.eps",
            "save_bool": False,
        },
    }

    # ------ other params ----------------------------------------------------------
    model = 'fake_model'
    global_settings = {'output_dir': tmpdir, 'experiment_name': experiment_name}
    db = 'dummy'
    random_seed = 1

    # Will be fixed in a subsequent MR
    rpvi_instance = None
    return rpvi_instance


def target_density(self, x=None, pdf=False):
    output_array = []
    mean = (np.array([0.5, 0.2, 0.6, 0.1, 0.2])).reshape(
        -1,
    )
    std_vec = np.array([0.1, 0.2, 0.01, 0.3, 0.1])
    cov = np.diag(std_vec**2)
    if pdf is False:
        for value in x:
            output_array.append(mvn.logpdf(value, mean=mean, cov=cov))
    else:
        for value in x:
            output_array.append(mvn.pdf(value, mean=mean, cov=cov))

    output_array = np.array(output_array)
    return output_array


@pytest.fixture()
def my_variational_distribution_obj():
    dimension = 5
    distribution_options = {
        "variational_family": "normal",
        "variational_approximation_type": "mean_field",
        "dimension": dimension,
    }
    my_variational_object = variational_inference_utils.create_variational_distribution(
        distribution_options
    )
    return my_variational_object


@pytest.fixture()
def visualization_obj(tmpdir):
    visualization_dict = {
        "method": {
            "method_options": {
                "result_description": {
                    "plotting_options": {
                        "plotting_dir": tmpdir,
                        "save_bool": False,
                        "plot_boolean": False,
                        "plot_name": "variat_params_convergence.eps",
                    }
                }
            }
        }
    }
    vis.from_config_create(visualization_dict)
