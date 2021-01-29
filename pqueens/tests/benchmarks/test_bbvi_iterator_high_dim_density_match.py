import os
import pickle
import pytest
import numpy as np
import pandas as pd
from mock import patch
from scipy.stats import multivariate_normal as mvn
from scipy.stats import entropy
from pqueens.utils import mcmc_utils
from pqueens.main import main
from pqueens.iterators.black_box_variational_bayes import BBVIIterator
from pqueens.utils import injector
import pqueens.visualization.variational_inference_visualization as vis


@pytest.mark.benchmark
def test_bbvi_density_match_high_dimensional(
    mocker,
    inputdir,
    tmpdir,
    my_variational_distribution_obj,
    my_likelihood_obj,
    target_distribution_obj,
    dummy_bbvi_instance,
    RV_dimension,
):
    # fix the random seed
    np.random.seed(1)

    ## Create the visualization_obj with the MLE estimates for plotting
    # visualization_obj(tmpdir, target_distribution_mean)

    # mock all parts of the algorithm that has to do with initialization or an underlying model
    mocker.patch(
        'pqueens.iterators.black_box_variational_bayes.BBVIIterator._initialize_prior_model',
        return_value=None,
    )
    mocker.patch(
        'pqueens.iterators.black_box_variational_bayes.BBVIIterator'
        '._initialize_variational_distribution',
        return_value=None,
    )

    # Function to patch get_log_posterior_unormalized with the initialized target distribution
    td = lambda self, x: target_density(self, target_distribution_obj, x=x, pdf=False)

    # Create the visualization_obj with the MLE estimates for plotting
    visualization_obj(tmpdir, target_distribution_obj.mean)

    # actual main call of bbvi with patched density for posterior
    with patch.object(BBVIIterator, 'get_log_posterior_unnormalized', td):
        # set some instance attributes that we need for out density matching test
        variational_params_cov = np.log(np.array(my_variational_distribution_obj.covariance)) / 2
        dummy_bbvi_instance.variational_params = my_variational_distribution_obj.mean + list(
            variational_params_cov
        )
        dummy_bbvi_instance.variational_params_array = np.empty(
            (len(dummy_bbvi_instance.variational_params), 0)
        )
        dummy_bbvi_instance.m_param_adams = np.zeros(
            (len(dummy_bbvi_instance.variational_params), 1)
        )
        dummy_bbvi_instance.v_param_adams = np.zeros(
            (len(dummy_bbvi_instance.variational_params), 1)
        )
        # set the initial variational density as an attribute
        dummy_bbvi_instance.variational_distribution_obj = my_variational_distribution_obj
        # actual run of the algorithm
        dummy_bbvi_instance.run()

        variational_distr_obj = dummy_bbvi_instance.variational_distribution_obj
        opt_variational_params = np.array(dummy_bbvi_instance.variational_params)

    # Actual tests
    opt_variational_samples = variational_distr_obj.draw(opt_variational_params, 10000)
    variational_pdf = []
    for sample in opt_variational_samples:
        variational_pdf.append(variational_distr_obj.pdf(opt_variational_params, sample))
    variational_pdf = np.array(variational_pdf)
    target_pdf = target_density(
        'dummy', target_distribution_obj, x=opt_variational_samples, pdf=True,
    ).flatten()
    kl_divergence = entropy(variational_pdf, target_pdf)
    assert kl_divergence < 5.0


@pytest.fixture()
def dummy_bbvi_instance(tmpdir, RV_dimension):
    #  ----- interesting params one might want to change ---------------------------
    n_samples_per_iter = 30
    min_requ_relative_change_variational_params = 0.001
    num_variables = RV_dimension
    learning_rate = 0.1
    # ------ params we want to keep fixed -----------------------------------------
    variational_transformation = None
    variational_family = 'normal'
    variational_approximation_type = 'mean_field'
    experiment_name = 'density_match'
    likelihood_model = {"type": "gaussian_static", "nugget_noise_factor": 1.3}
    result_description = {
        "write_results": False,
        "plotting_options": {
            "plot_booleans": [False, False],
            "plotting_dir": tmpdir,
            "plot_names": ["pdfs_params.eps", "variat_params_convergence.eps"],
            "save_bool": [False, False],
        },
    }

    # ------ other params ----------------------------------------------------------
    model = 'fake_model'
    global_settings = {'output_dir': tmpdir, 'experiment_name': experiment_name}
    db = 'dummy'
    random_seed = 1
    max_feval = 50000

    bbvi_instance = BBVIIterator(
        global_settings,
        model,
        result_description,
        db,
        experiment_name,
        min_requ_relative_change_variational_params,
        variational_family,
        variational_approximation_type,
        learning_rate,
        n_samples_per_iter,
        variational_transformation,
        random_seed,
        max_feval,
        num_variables,
    )
    return bbvi_instance


def target_density(self, target_distribution_obj, x=None, pdf=False):
    output_array = []
    if pdf is False:
        for value in x:
            output_array.append(target_distribution_obj.logpdf(value))
    else:
        for value in x:
            output_array.append(target_distribution_obj.pdf(value))

    output_array = np.array(output_array).reshape(1, -1)
    return output_array


@pytest.fixture()
def RV_dimension():
    RV_dimension = 100
    return RV_dimension


@pytest.fixture()
def target_distribution_obj(RV_dimension):
    # Initializing the target distribution
    mean = np.random.rand(RV_dimension) * RV_dimension / 10.0
    std = np.random.rand(RV_dimension) + 0.01

    distribution_options = {
        "distribution": "normal",
        "distribution_parameter": {"mean": mean, "standard_deviation": np.diag(std.tolist())},
    }
    target_distribution_object = mcmc_utils.create_proposal_distribution(distribution_options)
    return target_distribution_object


@pytest.fixture()
def my_variational_distribution_obj(RV_dimension):
    # Initializing the variational distribution
    mean = np.random.rand(RV_dimension) * RV_dimension / 10.0
    std = mean * 0.1

    distribution_options = {
        "distribution": "mean_field_normal",
        "distribution_parameter": {"mean": mean.tolist(), "standard_deviation": std.tolist()},
    }
    my_variational_object = mcmc_utils.create_proposal_distribution(distribution_options)
    return my_variational_object


@pytest.fixture()
def my_likelihood_obj():
    # dummy data
    y_obs = np.array([1, 1])
    my_likelihood_obj = mcmc_utils.GaussianLikelihood(y_obs, 1.0)
    return my_likelihood_obj


def visualization_obj(tmpdir, mle):
    # Keep in mind this can be slow due to the high dimension of the problem
    visualization_dict = {
        "method": {
            "method_options": {
                "result_description": {
                    "plotting_options": {
                        "plotting_dir": tmpdir,
                        "save_bool": [False, False],
                        "plot_booleans": [False, False],
                        "plot_names": ["pdfs_params.eps", "variat_params_convergence.eps"],
                        "MLE_comparison": mle,
                    }
                }
            }
        }
    }
    vis.from_config_create(visualization_dict)
