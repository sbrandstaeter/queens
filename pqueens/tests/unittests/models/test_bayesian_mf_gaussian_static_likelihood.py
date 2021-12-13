"""Unittests for Bayesian multi-fidelity Gaussian likelihood function."""
import numpy as np
import pytest

from pqueens.interfaces.bmfia_interface import BmfiaInterface
from pqueens.iterators.bmfia_iterator import BMFIAIterator
from pqueens.models.likelihood_models.bayesian_mf_gaussian_static_likelihood import (
    BMFGaussianStaticModel,
)
from pqueens.models.simulation_model import SimulationModel


# ------------ fixtures and params ---------------
@pytest.fixture()
def result_description():
    """Fixture for a dummy result description."""
    description = {"write_results": True}
    return description


@pytest.fixture()
def global_settings():
    """Fixture for dummy global settings."""
    global_set = {'output_dir': 'dummyoutput', 'experiment_name': 'dummy_exp_name'}
    return global_set


@pytest.fixture()
def parameters():
    """Fixture for dummy parameters."""
    params = {
        "random_variables": {
            "x1": {"type": "FLOAT", "size": 1, "min": -2.0, "max": 2.0},
            "x2": {"type": "FLOAT", "size": 1, "min": -2.0, "max": 2.0},
        },
        "random_fields": {
            'random_inflow': {
                'type': 'FLOAT',
                'dimension': 1,
                'min': 0,
                'max': 1,
                'corrstruct': 'non_stationary_squared_exp',
                'corr_length': 0.08,
                'std_hyperparam_rf': 0.1,
                'mean_fun': 'inflow_parabola',
                'mean_fun_params': [1.5],
                'num_points': 10,
            }
        },
    }
    return params


@pytest.fixture()
def dummy_model(parameters):
    """Fixture for dummy model."""
    model_name = 'dummy'
    interface = 'my_dummy_interface'
    model_parameters = parameters
    model = SimulationModel(model_name, interface, model_parameters)
    model.response = {'mean': 1.0}
    return model


@pytest.fixture()
def config():
    """Fixture for dummy configuration."""
    config = {
        "joint_density_approx": {
            "type": "gp_approximation_gpy",
            "features_config": "opt_features",
            "num_features": 1,
            "X_cols": 1,
        }
    }
    return config


@pytest.fixture()
def approximation_name():
    """Dummy approximation name for testing."""
    name = 'joint_density_approx'
    return name


@pytest.fixture()
def default_interface(config, approximation_name):
    """Dummy BMFIA interface for testing."""
    interface = BmfiaInterface(config, approximation_name)
    return interface


@pytest.fixture()
def settings_probab_mapping(config, approximation_name):
    """Dummy settings for the probabilistic mapping for testing."""
    settings = config[approximation_name]
    return settings


@pytest.fixture()
def default_bmfia_iterator(result_description, global_settings, dummy_model):
    """Dummy iterator for testing."""
    result_description = result_description
    global_settings = global_settings
    features_config = 'no_features'
    hf_model = 'dummy_hf_model'
    lf_model = 'dummy_lf_model'
    output_label = ['y']
    coord_labels = ['x_1', 'x_2']
    settings_probab_mapping = {'features_config': 'no_features'}
    db = 'dummy_db'
    external_geometry_obj = None
    x_train = np.array([[1, 2], [3, 4]])
    Y_LF_train = np.array([[2], [3]])
    Y_HF_train = np.array([[2.2], [3.3]])
    Z_train = np.array([[4], [5]])
    coords_experimental_data = np.array([[1, 2], [3, 4]])
    time_vec = np.array([1, 3])
    y_obs_vec = np.array([[2.1], [3.1]])
    gammas_train = None
    scaler_gamma = None

    iterator = BMFIAIterator(
        result_description,
        global_settings,
        features_config,
        hf_model,
        lf_model,
        output_label,
        coord_labels,
        settings_probab_mapping,
        db,
        external_geometry_obj,
        x_train,
        Y_LF_train,
        Y_HF_train,
        Z_train,
        coords_experimental_data,
        time_vec,
        y_obs_vec,
        gammas_train,
        scaler_gamma,
    )

    return iterator


@pytest.fixture()
def default_mf_likelihood(
    dummy_model, parameters, default_interface, settings_probab_mapping, default_bmfia_iterator
):
    """Default multi-fidelity Gaussian likelihood object that is used to test
    the methods of the object."""

    model_parameters = parameters
    nugget_noise_var = 0.1
    forward_model = dummy_model
    coords_mat = np.array([[1, 2], [3, 4]])
    time_vec = np.array([1, 2, 3, 4])
    y_obs_vec = np.array([[1, 2], [3, 4]])
    likelihood_noise_type = 'fixed'
    fixed_likelihood_noise_value = 0.1
    output_label = ['a', 'b']
    coord_labels = ['c', 'd']
    settings_probab_mapping = settings_probab_mapping
    mf_interface = default_interface
    bmfia_subiterator = default_bmfia_iterator
    noise_upper_bound = 0.1
    model_name = 'bmfia_model'
    x_train = None
    y_hf_train = None
    y_lfs_train = None
    gammas_train = None
    z_train = None
    eigenfunc_random_fields = None
    eigenvals = None
    f_mean_train = None
    noise_var = None
    noise_var_lst = []

    mf_likelihood = BMFGaussianStaticModel(
        model_name,
        model_parameters,
        nugget_noise_var,
        forward_model,
        coords_mat,
        time_vec,
        y_obs_vec,
        likelihood_noise_type,
        fixed_likelihood_noise_value,
        output_label,
        coord_labels,
        settings_probab_mapping,
        mf_interface,
        bmfia_subiterator,
        noise_upper_bound,
        x_train,
        y_hf_train,
        y_lfs_train,
        gammas_train,
        z_train,
        eigenfunc_random_fields,
        eigenvals,
        f_mean_train,
        noise_var,
        noise_var_lst,
    )
    return mf_likelihood


# ------------ unittests -------------------------
def test_init(
    dummy_model, parameters, default_interface, settings_probab_mapping, default_iterator
):
    """Test the init method of the multi-fidelity Gaussian likelihood
    function."""

    model_parameters = parameters
    nugget_noise_var = 0.1
    forward_model = dummy_model
    coords_mat = np.array([[1, 2], [3, 4]])
    time_vec = np.array([1, 2, 3, 4])
    y_obs_vec = np.array([[1], [3]])
    likelihood_noise_type = 'fixed'
    fixed_likelihood_noise_value = 0.1
    output_label = ['a', 'b']
    coord_labels = ['c', 'd']
    settings_probab_mapping = settings_probab_mapping
    mf_interface = default_interface
    bmfia_subiterator = default_iterator
    noise_upper_bound = 0.1
    model_name = 'bmfia_model'
    x_train = None
    y_hf_train = None
    y_lfs_train = None
    gammas_train = None
    z_train = None
    eigenfunc_random_fields = None
    eigenvals = None
    f_mean_train = None
    noise_var = None
    noise_var_lst = []

    model = BMFGaussianStaticModel(
        model_name,
        model_parameters,
        nugget_noise_var,
        forward_model,
        coords_mat,
        time_vec,
        y_obs_vec,
        likelihood_noise_type,
        fixed_likelihood_noise_value,
        output_label,
        coord_labels,
        settings_probab_mapping,
        mf_interface,
        bmfia_subiterator,
        noise_upper_bound,
        x_train,
        y_hf_train,
        y_lfs_train,
        gammas_train,
        z_train,
        eigenfunc_random_fields,
        eigenvals,
        f_mean_train,
        noise_var,
        noise_var_lst,
    )

    # tests / asserts ----------------------------------
    assert model.name == model_name
    assert model.uncertain_parameters == model_parameters
    assert model.forward_model == forward_model
    np.testing.assert_array_equal(model.coords_mat, coords_mat)
    np.testing.assert_array_equal(model.time_vec, time_vec)
    np.testing.assert_array_equal(model.y_obs_vec, y_obs_vec)
    assert model.output_label == output_label
    assert model.coord_labels == coord_labels

    assert model.mf_interface == mf_interface
    assert model.settings_probab_mapping == settings_probab_mapping
    assert model.x_train is None
    assert model.y_hf_train is None
    assert model.y_lfs_train is None
    assert model.gammas_train is None
    assert model.z_train is None
    assert model.eigenfunc_random_fields is None
    assert model.eigenvals is None
    assert model.f_mean_train is None
    assert model.bmfia_subiterator == bmfia_subiterator
    assert model.uncertain_parameters == model_parameters
    assert model.noise_var is None
    assert model.nugget_noise_var == nugget_noise_var
    assert model.likelihood_noise_type == likelihood_noise_type
    assert model.fixed_likelihood_noise_value == fixed_likelihood_noise_value
    assert model.noise_upper_bound == noise_upper_bound
    assert model.noise_var_lst == []


def test_evaluate_scalar(default_mf_likelihood, mocker):
    """Test the evaluate method by comparing the return value with the expected
    value and using a single point."""
    # pylint: disable=line-too-long
    mocker.patch(
        'pqueens.iterators.bayesian_mf_gaussian_static_likelihood.BMFGaussianStaticModel.eval_model',
        return_value=None,
    )
    # pylint: enable=line-too-long
    mf_log_likelihood = default_mf_likelihood.evaluate()
    breakpoint()
    expected_mf_log_lik = np.array([0])
    np.testing.assert_array_equal(expected_mf_log_lik, mf_log_likelihood)


def test_evaluate_mf_likelihood():
    pass


def test_log_likelihood_fun():
    pass


def test_get_feature_mat():
    pass


def test_initialize():
    pass


def test_build_approximation():
    pass


def test_input_dim_red():
    pass


def test_get_random_fields_and_truncated_basis():
    pass


def test_update_and_evaluate_forward_model():
    pass


def test_project_samples_on_truncated_basis():
    pass
