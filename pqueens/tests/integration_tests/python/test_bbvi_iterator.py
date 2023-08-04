"""Integration tests for the BBVI iterator."""
import pickle
from collections import namedtuple

import numpy as np
import pytest
from mock import Mock, patch
from scipy.stats import multivariate_normal as mvn

from pqueens.global_settings import GlobalSettings
from pqueens.iterators.black_box_variational_bayes import BBVIIterator
from pqueens.main import run
from pqueens.utils import injector
from pqueens.utils.stochastic_optimizer import Adam


def test_bbvi_density_match(
    mocker,
    inputdir,
    tmp_path,
    dummy_bbvi_instance,
):
    """Matching a Gaussian distribution."""
    # fix the random seed
    np.random.seed(1)

    # mock all parts of the algorithm that has to do with initialization or an underlying model
    mocker.patch(
        "pqueens.iterators.black_box_variational_bayes.BBVIIterator.pre_run",
        return_value=None,
    )

    # actual main call of bbvi with patched density for posterior
    with patch.object(BBVIIterator, 'get_log_posterior_unnormalized', target_density):
        variational_distr_obj = dummy_bbvi_instance.variational_distribution_obj
        mu = np.array([0.1, 0.7, 0.2, 0.3, 0.25])
        cov = np.exp(np.diag([0.5, 0.5, 0.5, 0.5, 0.5]) * 2)
        var_params = variational_distr_obj.construct_variational_parameters(mu, cov)
        dummy_bbvi_instance.variational_params = var_params
        dummy_bbvi_instance.stochastic_optimizer.set_gradient_function(
            dummy_bbvi_instance.get_gradient_function()
        )
        dummy_bbvi_instance.stochastic_optimizer.current_variational_parameters = (
            var_params.reshape(-1, 1)
        )  # actual run of the algorithm
        dummy_bbvi_instance.noise_list = [6, 6, 6]
        dummy_bbvi_instance.run()

        opt_variational_params = np.array(dummy_bbvi_instance.variational_params)

        elbo = dummy_bbvi_instance.iteration_data.elbo

    # Actual tests
    opt_variational_samples = variational_distr_obj.draw(opt_variational_params, 10000)
    variational_logpdf = variational_distr_obj.logpdf(
        opt_variational_params, opt_variational_samples
    )
    target_logpdf = target_density("dummy", x=opt_variational_samples, pdf=False).flatten()
    kl_divergence = np.abs(np.mean(variational_logpdf - target_logpdf))
    # Test if KL divergence is not too large
    assert kl_divergence < 10**5
    # Test if the elbo declined on average
    assert np.mean(elbo[-3:]) > np.mean(elbo[:3])


def test_bbvi_iterator_park91a_hifi(
    inputdir, tmp_path, create_experimental_data_park91a_hifi_on_grid
):
    """Test for the bbvi iterator based on the *park91a_hifi* function."""
    # generate json input file from template
    template = inputdir / "bbvi_park91a_hifi_template.yml"
    experimental_data_path = tmp_path
    plot_dir = tmp_path
    dir_dict = {
        "experimental_data_path": experimental_data_path,
        "plot_dir": plot_dir,
    }
    input_file = tmp_path / "bbvi_park91a_hifi.yml"
    injector.inject(dir_dict, template, input_file)

    # This seed is fixed so that the variational distribution is initialized so that the park
    # function can be evaluated correctly
    np.random.seed(211)

    # run the main routine of QUEENS
    run(input_file, tmp_path)

    # get the results of the QUEENS run
    result_file = tmp_path / "inverse_bbvi_park91a_hifi.pickle"
    with open(result_file, "rb") as handle:
        results = pickle.load(handle)
    elbo_list = results["iteration_data"]["elbo"]

    # Actual tests
    assert np.abs(results["variational_distribution"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.2
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5
    assert np.mean(elbo_list[-5:]) > np.mean(elbo_list[:5])


@pytest.fixture(name="dummy_bbvi_instance")
def fixture_dummy_bbvi_instance(tmp_path, my_variational_distribution):
    """Create visualization module."""
    #  ----- interesting params one might want to change ---------------------------
    n_samples_per_iter = 5
    max_feval = 10 * n_samples_per_iter
    num_variables = 5
    memory = 10
    natural_gradient_bool = False
    fim_dampening_bool = True
    variational_params_initialization_approach = "random"
    fim_decay_start_iter = 50
    fim_dampening_coefficient = 1e-2
    fim_dampening_lower_bound = 1e-8
    control_variates_scaling_type = "averaged"
    loo_cv_bool = False
    # ------ params we want to keep fixed -----------------------------------------
    variational_transformation = None
    experiment_name = 'density_match'
    result_description = {
        "iterative_field_names": ["elbo"],
        "write_results": False,
        "plotting_options": {
            "plot_boolean": False,
            "plotting_dir": tmp_path,
            "plot_name": "variational_params_convergence.eps",
            "save_bool": False,
        },
    }
    stochastic_optimizer = Adam(
        learning_rate=0.01,
        optimization_type="max",
        rel_l1_change_threshold=-1,
        rel_l2_change_threshold=-1,
        max_iteration=10000000,
    )

    # ------ other params ----------------------------------------------------------
    model = namedtuple("model", "normal_distribution")(
        normal_distribution=namedtuple("normal_distribution", "covariance")(covariance=0)
    )
    global_settings = GlobalSettings(experiment_name, output_dir=tmp_path)
    random_seed = 1

    with global_settings:
        parameters = Mock()
        parameters.num_parameters = num_variables

        bbvi_instance = BBVIIterator(
            model=model,
            parameters=parameters,
            result_description=result_description,
            variational_parameter_initialization=variational_params_initialization_approach,
            n_samples_per_iter=n_samples_per_iter,
            variational_transformation=variational_transformation,
            random_seed=random_seed,
            max_feval=max_feval,
            memory=memory,
            natural_gradient=natural_gradient_bool,
            FIM_dampening=fim_dampening_bool,
            decay_start_iteration=fim_decay_start_iter,
            dampening_coefficient=fim_dampening_coefficient,
            FIM_dampening_lower_bound=fim_dampening_lower_bound,
            control_variates_scaling_type=control_variates_scaling_type,
            loo_control_variates_scaling=loo_cv_bool,
            variational_distribution=my_variational_distribution,
            stochastic_optimizer=stochastic_optimizer,
            model_eval_iteration_period=1,
            resample=True,
        )
    return bbvi_instance


def target_density(self, x=None, pdf=False):
    """Create visualization module."""
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


@pytest.fixture(name="my_variational_distribution")
def fixture_my_variational_distribution():
    """Create visualization module."""
    dimension = 5
    distribution_options = {
        "variational_family": "normal",
        "variational_approximation_type": "mean_field",
        "dimension": dimension,
    }
    return distribution_options
