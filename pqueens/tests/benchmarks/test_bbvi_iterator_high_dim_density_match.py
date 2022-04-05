"""High dimensional Gaussian density match case."""
import numpy as np
import pytest
from mock import patch
from scipy.stats import multivariate_normal as mvn

import pqueens.visualization.variational_inference_visualization as vis
from pqueens.iterators.black_box_variational_bayes import BBVIIterator
from pqueens.utils import mcmc_utils, variational_inference_utils
from pqueens.utils.stochastic_optimizer import StochasticOptimizer

# Needed here to ensure that the target density is always the same.
np.random.seed(666)


@pytest.mark.benchmark
def test_bbvi_density_match_high_dimensional(
    mocker,
    inputdir,
    tmpdir,
    my_variational_distribution_obj,
    target_distribution_obj,
    dummy_bbvi_instance,
    rv_dimension,
):
    """Matching a high dimensional Gaussian distribution."""
    # fix the random seed
    np.random.seed(1)

    # mock all parts of the algorithm that has to do with initialization or an underlying model
    mocker.patch(
        "pqueens.iterators.black_box_variational_bayes.BBVIIterator.initialize_run",
        return_value=None,
    )

    # Function to patch get_log_posterior_unormalized with the initialized target distribution
    td = lambda self, x: target_density(self, target_distribution_obj, x=x, pdf=False)

    # Create the visualization_obj with the MLE estimates for plotting
    visualization_obj(tmpdir)

    # actual main call of bbvi with patched density for posterior
    with patch.object(BBVIIterator, 'get_log_posterior_unnormalized', td):

        # set some instance attributes that we need for out density matching test
        var_params = (
            dummy_bbvi_instance.variational_distribution_obj.initialize_parameters_randomly()
        )
        var_params = np.zeros(var_params.shape)
        dummy_bbvi_instance.variational_params = var_params
        dummy_bbvi_instance.stochastic_optimizer.gradient = (
            dummy_bbvi_instance._get_gradient_function()
        )
        dummy_bbvi_instance.stochastic_optimizer.current_variational_parameters = (
            var_params.reshape(-1, 1)  # actual run of the algorithm
        )
        dummy_bbvi_instance.noise_list = [6, 6, 6]

        # actual run of the algorithm
        dummy_bbvi_instance.run()

        variational_distr_obj = dummy_bbvi_instance.variational_distribution_obj
        opt_variational_params = np.array(dummy_bbvi_instance.variational_params)
        elbo = dummy_bbvi_instance.elbo_list[-1]
    # Actual tests
    opt_variational_samples = variational_distr_obj.draw(opt_variational_params, 10000)
    variational_logpdf = variational_distr_obj.logpdf(
        opt_variational_params, opt_variational_samples
    )
    target_logpdf = target_density(
        "dummy", target_distribution_obj, x=opt_variational_samples, pdf=False
    ).flatten()
    kl_divergence = np.abs(np.mean(variational_logpdf - target_logpdf))
    assert kl_divergence < 5.0
    assert np.abs(elbo) < 1e-5


@pytest.fixture()
def dummy_bbvi_instance(tmpdir, rv_dimension, my_variational_distribution_obj):
    """Initialize BBVI instance."""
    #  ----- interesting params one might want to change ---------------------------
    n_samples_per_iter = 30
    num_variables = rv_dimension
    max_feval = 1e5
    memory = 10
    natural_gradient_bool = True
    fim_dampening_bool = True
    export_quantities_over_iter = False
    variational_params_initialization_approach = "random"
    fim_decay_start_iter = 50
    fim_dampening_coefficient = 1e-2
    fim_dampening_lower_bound = 1e-8
    control_variates_scaling_type = "averaged"
    loo_cv_bool = False
    resample = True
    model_eval_iteration_period = 1000
    # ------ params we want to keep fixed -----------------------------------------
    variational_transformation = None
    variational_family = 'normal'
    experiment_name = 'density_match'
    result_description = {
        "write_results": False,
        "plotting_options": {
            "plot_boolean": False,
            "plotting_dir": tmpdir,
            "plot_name": "variat_params_convergence.eps",
            "save_bool": False,
        },
    }
    optimizer_config = {
        "stochastic_optimizer": "Adam",
        "learning_rate": 0.1,
        "optimization_type": "max",
        "rel_L1_change_threshold": 1e-8,
        "rel_L2_change_threshold": 1e-8,
        "max_iter": 10000000,
    }
    stochastic_optimizer = StochasticOptimizer.from_config_create_optimizer(optimizer_config)
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
        variational_params_initialization_approach=variational_params_initialization_approach,
        n_samples_per_iter=n_samples_per_iter,
        variational_transformation=variational_transformation,
        random_seed=random_seed,
        max_feval=max_feval,
        num_variables=num_variables,
        memory=memory,
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
        stochastic_optimizer=stochastic_optimizer,
        model_eval_iteration_period=model_eval_iteration_period,
        resample=resample,
    )
    return bbvi_instance


def target_density(self, target_distribution_obj, x=None, pdf=False):
    """Function to mock get_log_posterior_unnormalized."""
    output_array = []
    if pdf is False:
        for value in x:
            output_array.append(target_distribution_obj.logpdf(value))
    else:
        for value in x:
            output_array.append(target_distribution_obj.pdf(value))

    output_array = np.array(output_array).T
    return output_array


@pytest.fixture()
def rv_dimension():
    """Dimension of target distribution."""
    rv_dimension = 100
    return rv_dimension


@pytest.fixture()
def target_distribution_obj(rv_dimension):
    """Target probabilistic model."""
    # Initializing the target distribution
    mean = np.random.rand(rv_dimension)
    std = np.random.rand(rv_dimension) + 0.01

    distribution_options = {
        "distribution": "normal",
        "distribution_parameter": [mean, np.diag(std**2)],
    }
    target_distribution_object = mcmc_utils.create_proposal_distribution(distribution_options)
    return target_distribution_object


@pytest.fixture()
def my_variational_distribution_obj(rv_dimension):
    """Variational distribution object."""
    # Initializing the variational distribution
    distribution_options = {
        "variational_family": "normal",
        "variational_approximation_type": "mean_field",
        "dimension": rv_dimension,
    }
    my_variational_object = variational_inference_utils.create_variational_distribution(
        distribution_options
    )
    return my_variational_object


def visualization_obj(tmpdir):
    """Create visualization module."""
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
