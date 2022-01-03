import numpy as np
import pytest
import scipy.stats
from mock import patch

import pqueens.visualization.variational_inference_visualization as vis
from pqueens.iterators.black_box_variational_bayes import BBVIIterator
from pqueens.main import main
from pqueens.utils.variational_inference_utils import create_variational_distribution


@pytest.mark.benchmark
def test_bbvi_GMM_density_match(
    mocker,
    inputdir,
    tmpdir,
    variational_distribution_obj,
    dummy_bbvi_instance,
    visualization_obj,
):
    # The test is done with a fixed number of iterations, since the convergence criteria are not
    # optimatl yet

    # fix the random seed
    np.random.seed(1)
    # mock all parts of the algorithm that has to do with initialization or an underlying model
    mocker.patch(
        "pqueens.iterators.black_box_variational_bayes.BBVIIterator.initialize_run",
        return_value=None,
    )

    # actual main call of bbvi with patched density for posterior
    with patch.object(BBVIIterator, 'get_log_posterior_unnormalized', negative_potential):
        variational_distr_obj = dummy_bbvi_instance.variational_distribution_obj
        var_params = variational_distr_obj.initialize_parameters_randomly()
        dummy_bbvi_instance.variational_params = var_params
        dummy_bbvi_instance.variational_params_array = np.empty((len(var_params), 0))
        # actual run of the algorithm
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
def dummy_bbvi_instance(tmpdir, variational_distribution_obj):
    #  ----- interesting params one might want to change ---------------------------
    n_samples_per_iter = 5
    # -1 indicates to run a fixed number of samples
    min_requ_relative_change_variational_params = -1
    max_feval = 3000
    num_variables = 5
    learning_rate = 0.01
    memory = 50
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
        variational_distribution_obj=variational_distribution_obj,
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


def negative_potential(self, x=None):
    """The unnormalized probabilistic model used in this test is proportional
    to exp(-U) where U is a potential. Hence the log_posterior_unnormalized is
    given by -U.

    It is the first potential in https://arxiv.org/pdf/1505.05770.pdf
    (rotated by 70 degrees)
    """
    theta = np.radians(70)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    x = np.dot(x, R.T)
    z_1, z_2 = x[:, 0], x[:, 1]
    norm = np.sqrt(z_1 ** 2 + z_2 ** 2)
    outer_term_1 = 0.5 * ((norm - 2) / 0.4) ** 2
    inner_term_1 = np.exp((-0.5 * ((z_1 - 2) / 0.6) ** 2))
    inner_term_2 = np.exp((-0.5 * ((z_1 + 2) / 0.6) ** 2))
    outer_term_2 = np.log(inner_term_1 + inner_term_2 + 1e-7)
    u = outer_term_1 - outer_term_2
    log_posterior_unnormalized = -u
    return log_posterior_unnormalized


@pytest.fixture()
def variational_distribution_obj():
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
