import pytest
import numpy as np
from mock import patch
from scipy.stats import multivariate_normal as mvn
from pqueens.utils import mcmc_utils
from pqueens.utils import variational_inference_utils
from pqueens.iterators.black_box_variational_bayes import BBVIIterator
import pqueens.visualization.variational_inference_visualization as vis


@pytest.mark.benchmark
def test_bbvi_density_match_high_dimensional(
    mocker,
    inputdir,
    tmpdir,
    my_variational_distribution_obj,
    target_distribution_obj,
    dummy_bbvi_instance,
    RV_dimension,
):
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
        dummy_bbvi_instance.variational_params = var_params
        dummy_bbvi_instance.variational_params_array = np.empty((len(var_params), 0))

        # actual run of the algorithm
        dummy_bbvi_instance.run()

        variational_distr_obj = dummy_bbvi_instance.variational_distribution_obj
        opt_variational_params = np.array(dummy_bbvi_instance.variational_params)
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


@pytest.fixture()
def dummy_bbvi_instance(tmpdir, RV_dimension, my_variational_distribution_obj):
    #  ----- interesting params one might want to change ---------------------------
    n_samples_per_iter = 30
    relative_change_variational_params = 0.001
    num_variables = RV_dimension
    learning_rate = 0.1
    max_feval = 1e5
    num_variables = 5
    memory = 10
    natural_gradient_bool = True
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
    variational_approximation_type = 'mean_field'
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
        min_requ_relative_change_variational_params=relative_change_variational_params,
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
        variational_approximation_type=variational_approximation_type,
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


def target_density(self, target_distribution_obj, x=None, pdf=False):
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
def RV_dimension():
    RV_dimension = 100
    return RV_dimension


@pytest.fixture()
def target_distribution_obj(RV_dimension):
    # Initializing the target distribution
    mean = np.random.rand(RV_dimension)
    std = np.random.rand(RV_dimension) + 0.01

    distribution_options = {
        "distribution": "normal",
        "distribution_parameter": [mean, np.diag(std ** 2)],
    }
    target_distribution_object = mcmc_utils.create_proposal_distribution(distribution_options)
    return target_distribution_object


@pytest.fixture()
def my_variational_distribution_obj(RV_dimension):
    # Initializing the variational distribution
    distribution_options = {
        "variational_family": "normal",
        "variational_approximation_type": "mean_field",
        "dimension": RV_dimension,
    }
    my_variational_object = variational_inference_utils.create_variational_distribution(
        distribution_options
    )
    return my_variational_object


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
