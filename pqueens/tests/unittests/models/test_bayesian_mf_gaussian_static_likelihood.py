"""Unittests for Bayesian multi-fidelity Gaussian likelihood function."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

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
            "x1": {
                "type": "FLOAT",
                "size": 1,
                "min": -2.0,
                "max": 2.0,
                "distribution": "uniform",
                "distribution_parameter": [-2, 2],
            },
            "x2": {
                "type": "FLOAT",
                "size": 1,
                "min": -2.0,
                "max": 2.0,
                "distribution": "uniform",
                "distribution_parameter": [-2, 2],
            },
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
    model
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
    scaler_gamma = StandardScaler

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
    """Default multi-fidelity Gaussian likelihood object."""
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
    x_train = np.array([[1, 2], [3, 4]])
    y_hf_train = None
    y_lfs_train = None
    gammas_train = None
    z_train = None
    eigenfunc_random_fields = None
    eigenvals = None
    f_mean_train = None
    noise_var = 0.1
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


class InstanceMock:
    """Mock class."""

    @staticmethod
    def plot(self, *args, **kwargs):
        """Mock plot function."""
        return 1


@pytest.fixture
def mock_visualization():
    """Mock visualization."""
    my_mock = InstanceMock()
    return my_mock


class InstanceMockModel:
    """Mock model."""

    def __init__(self):
        """Init method for mock model."""
        self.variables = 1

    def evaluate(self, *args, **kwargs):
        """Mock evaluate method."""
        return {"mean": 1}


@pytest.fixture
def mock_model():
    """Mock model fixture."""
    my_mock = InstanceMockModel()
    return my_mock


# ------------ unittests -------------------------
@pytest.mark.unit_tests
def test_init(
    dummy_model, parameters, default_interface, settings_probab_mapping, default_bmfia_iterator
):
    """Test the init of the multi-fidelity Gaussian likelihood function."""
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


@pytest.mark.unit_tests
def test_evaluate(default_mf_likelihood, mocker, default_bmfia_iterator):
    """Compare return value with the expected value using a single point."""
    mf_log_likelihood_exp = np.array([1, 2])
    y_lf_mat = np.array([[1, 2]])
    # pylint: disable=line-too-long
    mp1 = mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_static_likelihood.BMFGaussianStaticModel._initialize',
        return_value=None,
    )

    # on purpose transpose y_lf_mat here to check if this is wrong orientation is corrected
    mp2 = mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_static_likelihood.BMFGaussianStaticModel._update_and_evaluate_forward_model',
        return_value=y_lf_mat.T,
    )

    mp3 = mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_static_likelihood.BMFGaussianStaticModel._evaluate_mf_likelihood',
        return_value=mf_log_likelihood_exp,
    )

    # pylint: enable=line-too-long

    mf_log_likelihood = default_mf_likelihood.evaluate()

    # assert statements
    mp1.assert_called_once()
    mp2.assert_called_once()
    mp3.assert_called_once()
    np.testing.assert_array_equal(y_lf_mat, mp3.call_args[0][0])
    np.testing.assert_array_equal(mf_log_likelihood, mf_log_likelihood_exp)


@pytest.mark.unit_tests
def test_evaluate_mf_likelihood(default_mf_likelihood, mocker):
    """Test the evaluation of the log multi-fidelity Gaussian likelihood."""
    # --- define some vectors and matrices -----
    y_lf_mat = np.array(
        [[1, 1, 1], [2, 2, 2]]
    )  # three dim output per point x in x_batch (row-wise)
    x_batch = np.array([[0, 0], [0, 1]])  # make points have distance 1
    diff_mat = np.array([[1, 1, 1], [2, 2, 2]])  # for simplicity we assume diff_mat equals
    var_y_mat = np.array([[1, 1, 1], [1, 1, 1]])

    # pylint: disable=line-too-long
    mp1 = mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_static_likelihood.BMFGaussianStaticModel._calculate_distance_vector_and_var_y',
        return_value=(diff_mat, var_y_mat),
    )
    mp2 = mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_static_likelihood.BMFGaussianStaticModel._calculate_likelihood_noise_var',
        return_value=None,
    )
    mp3 = mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_static_likelihood.BMFGaussianStaticModel._log_likelihood_fun',
        return_value=1,
    )
    # pylint: enable=line-too-long

    log_lik_mf = default_mf_likelihood._evaluate_mf_likelihood(y_lf_mat, x_batch)

    # ------ assert and test statements ------------------------------------
    mp1.assert_called_once()
    np.testing.assert_array_equal(y_lf_mat, mp1.call_args[0][0])
    np.testing.assert_array_equal(x_batch, mp1.call_args[0][1])

    mp2.assert_called_once()
    np.testing.assert_array_equal(diff_mat, mp2.call_args[0][0])

    np.testing.assert_equal(mp3.call_count, diff_mat.shape[0])
    np.testing.assert_array_equal(log_lik_mf, np.array([[1], [1]]))


@pytest.mark.unit_tests
def test_calculate_distance_vector_and_var_y(default_mf_likelihood, mocker):
    """Test the calculation of the distance vector."""
    x_batch = np.array([[0, 0], [0, 1]])  # make points have distance 1
    y_lf_mat = np.array([[1, 1], [2, 2]])  # three dim output per point x in x_batch (row-wise)
    z_mat = np.array([[1, 2], [3, 4]])
    m_f_mat = np.array([[2, 2], [3, 3]])
    var_y_mat_exp = np.array([[7, 7], [7, 7]])
    coords_mat = default_mf_likelihood.coords_mat[: y_lf_mat.shape[0]]
    diff_mat_exp = np.array([[1, 0], [0, -1]])

    # pylint: disable=line-too-long
    mp1 = mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_static_likelihood.BMFGaussianStaticModel._get_feature_mat',
        return_value=z_mat,
    )
    mp2 = mocker.patch(
        'pqueens.interfaces.bmfia_interface.BmfiaInterface.map',
        return_value=(m_f_mat, var_y_mat_exp),
    )
    # pylint: enable=line-too-long

    diff_mat_out, var_y_mat_out = default_mf_likelihood._calculate_distance_vector_and_var_y(
        y_lf_mat, x_batch
    )

    # --- assert statements and tests --------------------------
    mp1.assert_called_once()
    np.testing.assert_equal(mp1.call_args[0][0], y_lf_mat)
    np.testing.assert_equal(mp1.call_args[0][1], x_batch)
    np.testing.assert_equal(mp1.call_args[0][2], coords_mat)

    mp2.assert_called_once()
    np.testing.assert_equal(mp2.call_args[0][0], z_mat)

    np.testing.assert_equal(diff_mat_exp, diff_mat_out)
    np.testing.assert_equal(var_y_mat_exp, var_y_mat_out)


@pytest.mark.unit_tests
def test_calculate_likelihood_noise_var(default_mf_likelihood):
    """Test for (iterative) calculation of likelihood noise variance."""
    diff_mat = np.array([[1, 2], [2, 3]])

    default_mf_likelihood.likelihood_noise_type = "fixed"
    default_mf_likelihood._calculate_likelihood_noise_var(diff_mat)
    assert default_mf_likelihood.noise_var == default_mf_likelihood.fixed_likelihood_noise_value

    default_mf_likelihood.likelihood_noise_type = "jeffreys_prior"
    default_mf_likelihood._calculate_likelihood_noise_var(diff_mat)
    assert default_mf_likelihood.noise_var == 3.6


@pytest.mark.unit_tests
def test_calculate_rkhs_inner_product(default_mf_likelihood):
    """Test the calculation of the inner product and the differnt cases."""
    with pytest.raises(AssertionError):
        diff_vec = np.array([[1], [2]])
        inv_k_mf_mat = np.array([[1], [1]])
        inner_prod_rkhs = default_mf_likelihood._calculate_rkhs_inner_prod(diff_vec, inv_k_mf_mat)

    with pytest.raises(AssertionError):
        diff_vec = np.array([[1, 2]])
        inv_k_mf_mat = np.array([1])
        inner_prod_rkhs = default_mf_likelihood._calculate_rkhs_inner_prod(diff_vec, inv_k_mf_mat)

    diff_vec = np.array([[1, 2]])
    inv_k_mf_mat = np.array([[1]])
    inner_prod_rkhs = default_mf_likelihood._calculate_rkhs_inner_prod(diff_vec, inv_k_mf_mat)
    assert inner_prod_rkhs == 5

    diff_vec = np.array([[1, 2]])
    inv_k_mf_mat = np.array([[1, 1]])
    inner_prod_rkhs = default_mf_likelihood._calculate_rkhs_inner_prod(diff_vec, inv_k_mf_mat)
    assert inner_prod_rkhs == 5

    with pytest.raises(ValueError):
        diff_vec = np.array([[1, 2]])
        inv_k_mf_mat = np.array([[1], [2]])
        inner_prod_rkhs = default_mf_likelihood._calculate_rkhs_inner_prod(diff_vec, inv_k_mf_mat)


@pytest.mark.unit_tests
def test_log_likelihood_fun(default_mf_likelihood):
    """Test the calculation of the actual multi-fidelity log-likelihood.

    At the moment we assume that we only consider the variance of
    vectorized output and not the entire covariance structure.
    """
    expected_fixed = np.array([[-69.302937]])
    expected_jeffreys = np.array([[-67.8051]])
    mf_variance_vec = np.array([[0.1, 0.1, 0.1]])
    diff_vec = np.array([[1, 2, 3]])

    # test with fixed likelihood variance
    default_mf_likelihood.likelihood_noise_type = "fixed"
    log_mf_lik = default_mf_likelihood._log_likelihood_fun(mf_variance_vec, diff_vec)
    np.testing.assert_array_almost_equal(log_mf_lik, expected_fixed, decimal=4)

    # test with jeffreys prior likelihood variance
    default_mf_likelihood.likelihood_noise_type = "jeffreys_prior"
    log_mf_lik = default_mf_likelihood._log_likelihood_fun(mf_variance_vec, diff_vec)
    np.testing.assert_array_almost_equal(log_mf_lik, expected_jeffreys, decimal=4)

    # test with wrong likelihood noise type
    default_mf_likelihood.likelihood_noise_type = "dummy"
    with pytest.raises(ValueError):
        log_mf_lik = default_mf_likelihood._log_likelihood_fun(mf_variance_vec, diff_vec)

    # test dimensions of incoming variables 1) mf_variance_vec dim is wrong
    default_mf_likelihood.likelihood_noise_type = "fixed"
    mf_variance_vec = np.array([1, 1, 1])
    diff_vec = np.array([[1, 1, 1]])
    with pytest.raises(AssertionError):
        log_mf_lik = default_mf_likelihood._log_likelihood_fun(mf_variance_vec, diff_vec)

    # test dimensions of incoming variables 2) diff_vec dim is wrong
    default_mf_likelihood.likelihood_noise_type = "fixed"
    mf_variance_vec = np.array([[1, 1, 1]])
    diff_vec = np.array([1, 1, 1])
    with pytest.raises(AssertionError):
        log_mf_lik = default_mf_likelihood._log_likelihood_fun(mf_variance_vec, diff_vec)

    # test dimensions of incoming variables 3) input has different size
    default_mf_likelihood.likelihood_noise_type = "fixed"
    mf_variance_vec = np.array([[1, 1, 1]])
    diff_vec = np.array([[1, 1]])
    with pytest.raises(AssertionError):
        log_mf_lik = default_mf_likelihood._log_likelihood_fun(mf_variance_vec, diff_vec)

    # test dimensions of incoming variables 4) mf_variance_vec column vec
    default_mf_likelihood.likelihood_noise_type = "fixed"
    mf_variance_vec = np.array([[1], [1], [1]])
    diff_vec = np.array([[1, 1, 1]])
    with pytest.raises(AssertionError):
        log_mf_lik = default_mf_likelihood._log_likelihood_fun(mf_variance_vec, diff_vec)

    # test dimensions of incoming variables 5) diff_vec column vec
    default_mf_likelihood.likelihood_noise_type = "fixed"
    mf_variance_vec = np.array([[1, 1, 1]])
    diff_vec = np.array([[1], [1], [1]])
    with pytest.raises(AssertionError):
        log_mf_lik = default_mf_likelihood._log_likelihood_fun(mf_variance_vec, diff_vec)


@pytest.mark.unit_tests
def test_get_feature_mat(default_mf_likelihood, mocker):
    """Test the generation of low fidelity informative feautres."""
    # test wrong input dimensions 1) of y_lf_mat
    y_lf_mat = np.array([1, 2, 3])
    x_mat = np.array([[4, 5, 6]])
    coords_mat = np.array([[7, 8, 9]])
    with pytest.raises(AssertionError):
        z_mat = default_mf_likelihood._get_feature_mat(y_lf_mat, x_mat, coords_mat)

    # test wrong input dimensions 2) of y_x_mat
    y_lf_mat = np.array([[1, 2, 3]])
    x_mat = np.array([4, 5, 6])
    coords_mat = np.array([[7, 8, 9]])

    with pytest.raises(AssertionError):
        z_mat = default_mf_likelihood._get_feature_mat(y_lf_mat, x_mat, coords_mat)

    # test wrong input dimensions 3) of coords_mat
    y_lf_mat = np.array([[1, 2, 3]])
    x_mat = np.array([[4, 5, 6]])
    coords_mat = np.array([7, 8, 9])

    with pytest.raises(AssertionError):
        z_mat = default_mf_likelihood._get_feature_mat(y_lf_mat, x_mat, coords_mat)

    # test wrong features_config
    y_lf_mat = np.array([[1, 2, 3]])
    x_mat = np.array([[4, 5, 6]])
    coords_mat = np.array([[7, 8, 9]])
    default_mf_likelihood.bmfia_subiterator.settings_probab_mapping["features_config"] = "dummy"
    with pytest.raises(IOError):
        z_mat = default_mf_likelihood._get_feature_mat(y_lf_mat, x_mat, coords_mat)

    # test man_features without specifing 'X_cols' --> KeyError
    default_mf_likelihood.bmfia_subiterator.settings_probab_mapping[
        "features_config"
    ] = "man_features"
    with pytest.raises(KeyError):
        z_mat = default_mf_likelihood._get_feature_mat(y_lf_mat, x_mat, coords_mat)

    # test man_features with correct configuration
    expected_z_mat = np.array(
        [[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[4, 4, 4], [4, 4, 4], [4, 4, 4]]]
    )
    y_lf_mat = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    x_mat = np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6]])
    coords_mat = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])

    mocker.patch(
        'sklearn.preprocessing.StandardScaler.transform', return_value=np.atleast_2d(x_mat[:, 0]).T,
    )

    default_mf_likelihood.bmfia_subiterator.settings_probab_mapping[
        "features_config"
    ] = "man_features"
    default_mf_likelihood.bmfia_subiterator.settings_probab_mapping['X_cols'] = 0
    z_mat = default_mf_likelihood._get_feature_mat(y_lf_mat, x_mat, coords_mat)
    np.testing.assert_array_almost_equal(z_mat, expected_z_mat, decimal=4)

    # test opt_features
    default_mf_likelihood.bmfia_subiterator.settings_probab_mapping[
        "features_config"
    ] = "opt_features"
    default_mf_likelihood.bmfia_subiterator.settings_probab_mapping["num_features"] = 1
    with pytest.raises(NotImplementedError):
        z_mat = default_mf_likelihood._get_feature_mat(y_lf_mat, x_mat, coords_mat)

    # test opt_features with num features < 1 --> error
    default_mf_likelihood.bmfia_subiterator.settings_probab_mapping[
        "features_config"
    ] = "opt_features"
    default_mf_likelihood.bmfia_subiterator.settings_probab_mapping["num_features"] = 0
    with pytest.raises(AssertionError):
        z_mat = default_mf_likelihood._get_feature_mat(y_lf_mat, x_mat, coords_mat)

    # test opt_features with num features None --> error
    default_mf_likelihood.bmfia_subiterator.settings_probab_mapping[
        "features_config"
    ] = "opt_features"
    default_mf_likelihood.bmfia_subiterator.settings_probab_mapping["num_features"] = None
    with pytest.raises(AssertionError):
        z_mat = default_mf_likelihood._get_feature_mat(y_lf_mat, x_mat, coords_mat)

    # test coord_features without specifing 'coord_cols' --> KeyError
    default_mf_likelihood.bmfia_subiterator.settings_probab_mapping[
        "features_config"
    ] = "coord_features"
    with pytest.raises(KeyError):
        z_mat = default_mf_likelihood._get_feature_mat(y_lf_mat, x_mat, coords_mat)

    # test coord_features with correct configuration
    expected_z_mat = np.array(
        [[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[7, 7, 7], [10, 10, 10], [13, 13, 13]]]
    )
    default_mf_likelihood.bmfia_subiterator.settings_probab_mapping[
        "features_config"
    ] = "coord_features"
    default_mf_likelihood.bmfia_subiterator.settings_probab_mapping['coords_cols'] = 0
    z_mat = default_mf_likelihood._get_feature_mat(y_lf_mat, x_mat, coords_mat)
    np.testing.assert_array_almost_equal(z_mat, expected_z_mat, decimal=4)

    # test no_features
    expected_z_mat = y_lf_mat
    default_mf_likelihood.bmfia_subiterator.settings_probab_mapping[
        "features_config"
    ] = "no_features"
    z_mat = default_mf_likelihood._get_feature_mat(y_lf_mat, x_mat, coords_mat)
    np.testing.assert_array_almost_equal(z_mat, expected_z_mat, decimal=4)

    # test time_features
    expected_z_mat = np.array([[1, 2, 3, 1], [1, 2, 3, 2.5], [1, 2, 3, 4]])
    default_mf_likelihood.bmfia_subiterator.settings_probab_mapping[
        "features_config"
    ] = "time_features"
    default_mf_likelihood.time_vec = np.linspace(0, 10, y_lf_mat.shape[1])
    z_mat = default_mf_likelihood._get_feature_mat(y_lf_mat, x_mat, coords_mat)
    np.testing.assert_array_almost_equal(z_mat, expected_z_mat, decimal=4)


@pytest.mark.unit_tests
def test_initialize(default_mf_likelihood, mocker):
    """Test the initialization of the mf likelihood model."""
    coords_mat = np.array([[1, 2, 3], [2, 2, 2]])
    time_vec = np.linspace(1, 10, 3)
    y_obs_vec = np.array([[5, 5, 5], [6, 6, 6]])

    default_mf_likelihood.coords_mat = coords_mat
    default_mf_likelihood.time_vec = time_vec
    default_mf_likelihood.y_obs_vec = y_obs_vec

    # pylint: disable=line-too-long
    mo_1 = mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_static_likelihood.print_bmfia_acceleration',
        return_value=None,
    )
    mo_2 = mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_static_likelihood.BMFGaussianStaticModel._build_approximation',
        return_value=None,
    )
    # pylint: enable=line-too-long
    default_mf_likelihood._initialize()

    # actual tests / asserts
    mo_1.assert_called_once()
    mo_2.assert_called_once()
    np.testing.assert_array_almost_equal(
        default_mf_likelihood.bmfia_subiterator.coords_experimental_data, coords_mat, decimal=4
    )
    np.testing.assert_array_almost_equal(
        default_mf_likelihood.bmfia_subiterator.time_vec, time_vec, decimal=4
    )
    np.testing.assert_array_almost_equal(
        default_mf_likelihood.bmfia_subiterator.y_obs_vec, y_obs_vec, decimal=4
    )


@pytest.mark.unit_tests
def test_build_approximation(default_mf_likelihood, mocker):
    """Test for the build stage of the probabilistic regression model."""
    z_train = np.array([[1, 1, 1], [2, 2, 2]])
    y_hf_train = np.array([[1, 1], [2, 2]])

    # pylint: disable=line-too-long
    mo_1 = mocker.patch(
        'pqueens.iterators.bmfia_iterator.BMFIAIterator.core_run',
        return_value=(z_train, y_hf_train),
    )
    mo_2 = mocker.patch(
        'pqueens.interfaces.bmfia_interface.BmfiaInterface.build_approximation', return_value=None,
    )
    mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_static_likelihood.qvis.bmfia_visualization_instance',
        return_value=mock_visualization,
    )
    mo_4 = mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_static_likelihood.qvis.bmfia_visualization_instance.plot',
    )
    # pylint: enable=line-too-long

    default_mf_likelihood._build_approximation()

    # actual asserts/tests
    mo_1.assert_called_once()
    mo_2.assert_called_once()
    np.testing.assert_array_almost_equal(mo_2.call_args[0][0], default_mf_likelihood.z_train)
    np.testing.assert_array_almost_equal(mo_2.call_args[0][1], default_mf_likelihood.y_hf_train)
    mo_4.assert_called_once()
    np.testing.assert_array_almost_equal(mo_4.call_args[0][0], default_mf_likelihood.z_train)
    np.testing.assert_array_almost_equal(mo_4.call_args[0][1], default_mf_likelihood.y_hf_train)
    np.testing.assert_array_almost_equal(
        mo_4.call_args[0][2], default_mf_likelihood.mf_interface.probabilistic_mapping_obj_lst
    )


@pytest.mark.unit_tests
def test_input_dim_red(default_mf_likelihood, mocker):
    """Test for the input dimensionality reduction routine."""
    # pylint: disable=line-too-long
    mo_1 = mocker.patch(
        "pqueens.models.likelihood_models.bayesian_mf_gaussian_static_likelihood.BMFGaussianStaticModel.input_dim_red"
    )
    # pylint: enable=line-too-long
    default_mf_likelihood.input_dim_red()
    mo_1.assert_called_once()


@pytest.mark.unit_tests
def test_get_random_fields_and_truncated_basis(default_mf_likelihood):
    """Test the get-method for truncated random fields."""
    with pytest.raises(NotImplementedError):
        default_mf_likelihood.get_random_fields_and_truncated_basis()


@pytest.mark.unit_tests
def test_update_and_evaluate_forward_model(default_mf_likelihood, mock_model):
    """Test if forward model (lf model) is updated and evaluated correctly."""
    y_mat_expected = 1
    default_mf_likelihood.forward_model = mock_model
    y_mat = default_mf_likelihood._update_and_evaluate_forward_model()

    # actual tests / asserts
    np.testing.assert_array_almost_equal(y_mat, y_mat_expected, decimal=4)


@pytest.mark.unit_tests
def test_project_samples_on_truncated_basis(default_mf_likelihood, mocker):
    """Test projection of samples on the truncated basis for random fields."""
    expected_coefs_mat = np.array([[12, 18], [30, 45]])
    truncated_basis_dict = {
        "field_1": {
            "samples": np.array([[1, 2, 3], [4, 5, 6]]),
            "trunc_basis": np.array([[2, 2, 2], [3, 3, 3]]),
        }
    }
    num_samples = 2

    coefs_mat = default_mf_likelihood._project_samples_on_truncated_basis(
        truncated_basis_dict, num_samples
    )

    np.testing.assert_array_almost_equal(coefs_mat, expected_coefs_mat, decimal=4)
