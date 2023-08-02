"""Rezende potential matching case with GMM."""
import numpy as np
import pytest
from mock import patch

import pqueens.visualization.variational_inference_visualization as vis
from pqueens.iterators.black_box_variational_bayes import BBVIIterator
from pqueens.utils.variational_inference_utils import create_variational_distribution


def test_bbvi_GMM_density_match(
    mocker,
    variational_distribution_obj,
    dummy_bbvi_instance,
    visualization_obj,
):
    """Matching Rezende potential with GMM."""
    # The test is done with a fixed number of iterations, since the convergence criteria are not
    # optimal yet

    # fix the random seed
    np.random.seed(1)
    # mock all parts of the algorithm that has to do with initialization or an underlying model
    mocker.patch(
        "pqueens.iterators.black_box_variational_bayes.BBVIIterator.pre_run",
        return_value=None,
    )

    # actual main call of bbvi with patched density for posterior
    with patch.object(BBVIIterator, 'get_log_posterior_unnormalized', negative_potential):
        variational_distr_obj = dummy_bbvi_instance.variational_distribution_obj
        var_params = variational_distr_obj.initialize_parameters_randomly()
        dummy_bbvi_instance.variational_params = var_params
        dummy_bbvi_instance.stochastic_optimizer.set_gradient_function(
            dummy_bbvi_instance._get_gradient_function()
        )
        dummy_bbvi_instance.stochastic_optimizer.current_variational_parameters = (
            var_params.reshape(-1, 1)  # actual run of the algorithm
        )
        dummy_bbvi_instance.noise_list = [6, 6, 6]
        dummy_bbvi_instance.run()

        opt_variational_params = np.array(dummy_bbvi_instance.variational_params)

        elbo = dummy_bbvi_instance.elbo_list

    # Actual tests
    opt_variational_samples = variational_distr_obj.draw(opt_variational_params, 10000)
    variational_pdf = variational_distr_obj.pdf(opt_variational_params, opt_variational_samples)

    # Approximitely the exact pdf, the normalization constant was obtained using numerical
    # integration
    target_pdf = negative_potential('dummy', x=opt_variational_samples).flatten() - np.log(
        6.53715345
    )
    kl_divergence = np.abs(np.mean(variational_pdf - target_pdf))
    assert elbo[-1] > elbo[0]
    assert kl_divergence < 5.0


@pytest.fixture()
def dummy_bbvi_instance(tmp_path, variational_distribution_obj):
    """Initialize BBVI instance."""
    #  ----- interesting params one might want to change ---------------------------
    n_samples_per_iter = 5
    # -1 indicates to run a fixed number of samples
    max_feval = 3000
    num_variables = 5
    memory = 50
    natural_gradient_bool = True
    fim_dampening_bool = True
    export_quantities_over_iter = False
    variational_params_initialization_approach = "random"
    fim_decay_start_iter = 50
    fim_dampening_coefficient = 1e-2
    fim_dampening_lower_bound = 1e-8
    control_variates_scaling_type = "averaged"
    loo_cv_bool = False
    model_eval_iteration_period = 1
    resample = False
    # ------ params we want to keep fixed -----------------------------------------
    variational_transformation = None
    variational_family = 'normal'
    experiment_name = 'density_match'
    result_description = {
        "write_results": False,
        "plotting_options": {
            "plot_boolean": False,
            "plotting_dir": tmp_path,
            "plot_name": "variational_params_convergence.eps",
            "save_bool": False,
        },
    }
    optimizer_config = {
        "stochastic_optimizer": "adam",
        "learning_rate": 0.01,
        "optimization_type": "max",
        "rel_l1_change_threshold": -1,
        "rel_l2_change_threshold": -1,
        "max_iteration": 10000000,
    }
    stochastic_optimizer = from_config_create_optimizer(optimizer_config)
    # ------ other params ----------------------------------------------------------
    model = 'fake_model'
    global_settings = {'output_dir': tmp_path, 'experiment_name': experiment_name}
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
        variational_distribution_obj=variational_distribution_obj,
        variational_family=variational_family,
        stochastic_optimizer=stochastic_optimizer,
        model_eval_iteration_period=model_eval_iteration_period,
        resample=resample,
    )
    return bbvi_instance


def negative_potential(self, x=None):
    r"""Rezende potential.

    The unnormalized probabilistic model used in this test is proportional
    to  :math:`exp(-U)` where :math:`U` is a potential. Hence the *log_posterior_unnormalized*
    is given by :math:`-U`.

    It is the first potential in https://arxiv.org/pdf/1505.05770.pdf
    (rotated by 70 degrees).
    """
    theta = np.radians(70)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    x = np.dot(x, R.T)
    z_1, z_2 = x[:, 0], x[:, 1]
    norm = np.sqrt(z_1**2 + z_2**2)
    outer_term_1 = 0.5 * ((norm - 2) / 0.4) ** 2
    inner_term_1 = np.exp((-0.5 * ((z_1 - 2) / 0.6) ** 2))
    inner_term_2 = np.exp((-0.5 * ((z_1 + 2) / 0.6) ** 2))
    outer_term_2 = np.log(inner_term_1 + inner_term_2 + 1e-7)
    u = outer_term_1 - outer_term_2
    log_posterior_unnormalized = -u
    return log_posterior_unnormalized


@pytest.fixture()
def variational_distribution_obj():
    """Variational distribution object."""
    k = 4
    d = 2
    base_distribution = {
        "variational_family": "normal",
        "variational_approximation_type": "fullrank",
    }
    mixture_model_dict = {
        "variational_family": "mixture_model",
        "dimension": d,
        "num_components": k,
        "base_distribution": base_distribution,
    }
    mixture_model_obj = create_variational_distribution(mixture_model_dict)
    return mixture_model_obj


@pytest.fixture()
def visualization_obj(tmp_path):
    """Create visualization module."""
    visualization_dict = {
        "method": {
            "result_description": {
                "plotting_options": {
                    "plotting_dir": tmp_path,
                    "save_bool": False,
                    "plot_boolean": False,
                    "plot_name": "variat_params_convergence.eps",
                }
            }
        }
    }
    vis.from_config_create(visualization_dict)
