import os
import pickle

import numpy as np
import pandas as pd
import pytest
from mock import patch
from scipy.stats import multivariate_normal as mvn

import pqueens.visualization.variational_inference_visualization as vis
from pqueens.iterators.black_box_variational_bayes import BBVIIterator
from pqueens.main import main
from pqueens.tests.integration_tests.example_simulator_functions.park91a_hifi_coords import (
    park91a_hifi_coords,
)
from pqueens.utils import injector, variational_inference_utils


@pytest.mark.integration_tests
def test_bbvi_density_match(
    mocker,
    inputdir,
    tmpdir,
    dummy_bbvi_instance,
    visualization_obj,
):
    # fix the random seed
    np.random.seed(1)

    # mock all parts of the algorithm that has to do with initialization or an underlying model
    mocker.patch(
        "pqueens.iterators.black_box_variational_bayes.BBVIIterator.initialize_run",
        return_value=None,
    )

    # actual main call of bbvi with patched density for posterior
    with patch.object(BBVIIterator, 'get_log_posterior_unnormalized', target_density):
        variational_distr_obj = dummy_bbvi_instance.variational_distribution_obj
        mu = np.array([0.1, 0.7, 0.2, 0.3, 0.25])
        cov = np.exp(np.diag([0.5, 0.5, 0.5, 0.5, 0.5]) * 2)
        var_params = variational_distr_obj.construct_variational_params(mu, cov)
        dummy_bbvi_instance.variational_params = var_params
        dummy_bbvi_instance.variational_params_array = np.empty((len(var_params), 0))
        # actual run of the algorithm
        dummy_bbvi_instance.run()

        opt_variational_params = np.array(dummy_bbvi_instance.variational_params)

        elbo = dummy_bbvi_instance.elbo_list

    # Actual tests
    opt_variational_samples = variational_distr_obj.draw(opt_variational_params, 10000)
    variational_logpdf = variational_distr_obj.logpdf(
        opt_variational_params, opt_variational_samples
    )
    target_logpdf = target_density("dummy", x=opt_variational_samples, pdf=False).flatten()
    kl_divergence = np.abs(np.mean(variational_logpdf - target_logpdf))
    # Test if KL divergence is not too large
    assert kl_divergence < 10 ** 5
    # Test if the elbo declined
    assert elbo[0] < elbo[-1]


@pytest.mark.integration_tests
def test_bbvi_iterator_park91a_hifi(inputdir, tmpdir, design_and_write_experimental_data_to_csv):
    """Integration test for the bbvi iterator based on the park91a_hifi
    function."""

    # generate json input file from template
    template = os.path.join(inputdir, "bbvi_park91a_hifi_template.json")
    experimental_data_path = tmpdir
    plot_dir = tmpdir
    dir_dict = {
        "experimental_data_path": experimental_data_path,
        "plot_dir": plot_dir,
    }
    input_file = os.path.join(tmpdir, "bbvi_park91a_hifi.json")
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
    result_file = os.path.join(tmpdir, "inverse_bbvi_park91a_hifi.pickle")
    with open(result_file, "rb") as handle:
        results = pickle.load(handle)
    elbo_list = results["elbo"]
    # Actual tests
    assert np.abs(results["variational_distr"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distr"]["mean"][1] - 0.2) < 0.1
    assert results["variational_distr"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distr"]["covariance"][1, 1] ** 0.5 < 0.5
    assert np.mean(elbo_list[-5:]) > elbo_list[0]


@pytest.fixture()
def dummy_bbvi_instance(tmpdir, my_variational_distribution_obj):
    #  ----- interesting params one might want to change ---------------------------
    n_samples_per_iter = 5
    # -1 indicates to run a fixed number of samples
    min_requ_relative_change_variational_params = -1
    max_feval = 10 * n_samples_per_iter
    num_variables = 5
    learning_rate = 0.01
    memory = 10
    natural_gradient_bool = False
    clipping_bool = True
    fim_dampening_bool = True
    export_quantities_over_iter = False
    variational_params_initialization_approach = "random"
    num_iter_average_convergence = 5
    gradient_clipping_norm_threshold = 1e6
    fim_decay_start_iter = 50
    fim_dampening_coefficient = 1e-2
    fim_dampening_lower_bound = 1e-8
    control_variates_scaling_type = "averaged"
    loo_cv_bool = False
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

    bbvi_instance = BBVIIterator(
        global_settings=global_settings,
        model=model,
        result_description=result_description,
        db=db,
        experiment_name=experiment_name,
        min_requ_relative_change_variational_params=min_requ_relative_change_variational_params,
        variational_params_initialization_approach=variational_params_initialization_approach,
        n_samples_per_iter=n_samples_per_iter,
        variational_transformation=variational_transformation,
        random_seed=random_seed,
        max_feval=max_feval,
        num_iter_average_convergence=num_iter_average_convergence,
        num_variables=num_variables,
        memory=memory,
        learning_rate=learning_rate,
        clipping_bool=clipping_bool,
        gradient_clipping_norm_threshold=gradient_clipping_norm_threshold,
        natural_gradient_bool=natural_gradient_bool,
        fim_dampening_bool=fim_dampening_bool,
        fim_decay_start_iter=fim_decay_start_iter,
        fim_dampening_coefficient=fim_dampening_coefficient,
        fim_dampening_lower_bound=fim_dampening_lower_bound,
        export_quantities_over_iter=export_quantities_over_iter,
        control_variates_scaling_type=control_variates_scaling_type,
        loo_cv_bool=loo_cv_bool,
        variational_distribution_obj=my_variational_distribution_obj,
        variational_family=variational_family,
        optimization_iteration=0,
        v_param_adams=0,
        m_param_adams=0,
        n_sims=0,
        variational_params=0,
        f_mat=None,
        h_mat=None,
        grad_elbo=None,
        log_variational_mat=None,
        grad_params_log_variational_mat=None,
        log_posterior_unnormalized=None,
        prior_obj_list=None,
        elbo_list=[],
        samples_list=[],
        parameter_list=[],
        log_posterior_unnormalized_list=[],
        ess_list=[],
        noise_list=[0, 0],
        variational_params_array=None,
    )
    return bbvi_instance


def target_density(self, x=None, pdf=False):
    output_array = []
    mean = (np.array([0.5, 0.2, 0.6, 0.1, 0.2])).reshape(
        -1,
    )
    std_vec = np.array([0.1, 0.2, 0.01, 0.3, 0.1])
    cov = np.diag(std_vec ** 2)
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
def design_and_write_experimental_data_to_csv(tmpdir):
    # Fix random seed
    np.random.seed(seed=1)

    # create target inputs
    x1 = 0.5
    x2 = 0.2

    # use x3 and x4 as coordinates and create coordinate grid (same as in park91a_hifi_coords)
    xx3 = np.linspace(0, 1, 4)
    xx4 = np.linspace(0, 1, 4)
    x3_vec, x4_vec = np.meshgrid(xx3, xx4)
    x3_vec = x3_vec.flatten()
    x4_vec = x4_vec.flatten()

    # generate clean function output for fake test data
    y_vec = []
    for x3, x4 in zip(x3_vec, x4_vec):
        y_vec.append(park91a_hifi_coords(x1, x2, x3, x4))
    y_vec = np.array(y_vec)

    # add artificial noise to fake measurements
    sigma_n = 0.001
    noise_vec = np.random.normal(loc=0, scale=sigma_n, size=(y_vec.size,))
    y_fake = y_vec + noise_vec

    # write fake data to csv
    data_dict = {
        'x3': x3_vec,
        'x4': x4_vec,
        'y_obs': y_fake,
    }
    experimental_data_path = os.path.join(tmpdir, 'experimental_data.csv')
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)


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
