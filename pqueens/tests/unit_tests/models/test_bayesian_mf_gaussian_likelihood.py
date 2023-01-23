"""Unit tests for Bayesian multi-fidelity Gaussian likelihood function."""

import numpy as np
import pytest

from pqueens.interfaces.bmfia_interface import BmfiaInterface
from pqueens.iterators.bmfia_iterator import BMFIAIterator
from pqueens.models.likelihood_models.bayesian_mf_gaussian_likelihood import BMFGaussianModel
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
            "x1": {"dimension": 1, "distribution": "uniform", "lower_bound": -2, "upper_bound": 2},
            "x2": {"dimension": 1, "distribution": "uniform", "lower_bound": -2, "upper_bound": 2},
        },
    }
    return params


@pytest.fixture()
def dummy_model(parameters):
    """Fixture for dummy model."""
    model_name = 'dummy'
    interface = 'my_dummy_interface'
    model = SimulationModel(model_name, interface)
    return model


@pytest.fixture()
def config():
    """Fixture for dummy configuration."""
    config = {
        "joint_density_approx": {
            "type": "gp_approximation_gpy",
            "num_processors_multi_processing": 2,
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
    num_processors_multi_processing = 2
    interface = BmfiaInterface(config, approximation_name, num_processors_multi_processing)
    return interface


@pytest.fixture()
def settings_probab_mapping(config, approximation_name):
    """Dummy settings for the probabilistic mapping for testing."""
    settings = config[approximation_name]
    return settings


@pytest.fixture()
def default_bmfia_iterator(result_description, global_settings):
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
    x_train = np.array([[1, 2], [3, 4]])
    Y_LF_train = np.array([[2], [3]])
    Y_HF_train = np.array([[2.2], [3.3]])
    Z_train = np.array([[4], [5]])
    coords_experimental_data = np.array([[1, 2], [3, 4]])
    time_vec = np.array([1, 3])
    y_obs = np.array([[2.1], [3.1]])

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
        x_train,
        Y_LF_train,
        Y_HF_train,
        Z_train,
        coords_experimental_data,
        time_vec,
        y_obs,
    )

    return iterator


@pytest.fixture()
def default_mf_likelihood(
    dummy_model, parameters, default_interface, settings_probab_mapping, default_bmfia_iterator
):
    """Default multi-fidelity Gaussian likelihood object."""
    nugget_noise_var = 0.1
    forward_model = dummy_model
    coords_mat = np.array([[1, 2], [3, 4]])
    time_vec = np.array([1, 2, 3, 4])
    y_obs = np.array([[1, 2], [3, 4]])
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
    z_train = None
    eigenfunc_random_fields = None
    eigenvals = None
    f_mean_train = None
    noise_var = 0.1
    noise_var_lst = []

    mf_likelihood = BMFGaussianModel(
        model_name,
        nugget_noise_var,
        forward_model,
        coords_mat,
        time_vec,
        y_obs,
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


# ------------ unit_tests -------------------------
def test_init(
    dummy_model, parameters, default_interface, settings_probab_mapping, default_bmfia_iterator
):
    """Test the init of the multi-fidelity Gaussian likelihood function."""
    nugget_noise_var = 0.1
    forward_model = dummy_model
    coords_mat = np.array([[1, 2], [3, 4]])
    time_vec = np.array([1, 2, 3, 4])
    y_obs = np.array([[1], [3]])
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
    z_train = None
    eigenfunc_random_fields = None
    eigenvals = None
    f_mean_train = None
    noise_var = None
    noise_var_lst = []

    model = BMFGaussianModel(
        model_name,
        nugget_noise_var,
        forward_model,
        coords_mat,
        time_vec,
        y_obs,
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
        z_train,
        eigenfunc_random_fields,
        eigenvals,
        f_mean_train,
        noise_var,
        noise_var_lst,
    )

    # tests / asserts ----------------------------------
    assert model.name == model_name
    assert model.forward_model == forward_model
    np.testing.assert_array_equal(model.coords_mat, coords_mat)
    np.testing.assert_array_equal(model.time_vec, time_vec)
    np.testing.assert_array_equal(model.y_obs, y_obs)
    assert model.output_label == output_label
    assert model.coord_labels == coord_labels

    assert model.mf_interface == mf_interface
    assert model.settings_probab_mapping == settings_probab_mapping
    assert model.x_train is None
    assert model.y_hf_train is None
    assert model.y_lfs_train is None
    assert model.z_train is None
    assert model.eigenfunc_random_fields is None
    assert model.eigenvals is None
    assert model.f_mean_train is None
    assert model.bmfia_subiterator == bmfia_subiterator
    assert model.noise_var is None
    assert model.nugget_noise_var == nugget_noise_var
    assert model.likelihood_noise_type == likelihood_noise_type
    assert model.fixed_likelihood_noise_value == fixed_likelihood_noise_value
    assert model.noise_upper_bound == noise_upper_bound
    assert model.noise_var_lst == []


def test_evaluate(default_mf_likelihood, mocker, default_bmfia_iterator):
    """Compare return value with the expected value using a single point."""
    mf_log_likelihood_exp = np.array([1, 2])
    y_lf_mat = np.array([[1, 2]])
    # pylint: disable=line-too-long
    mp1 = mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_likelihood.BMFGaussianModel._initialize',
        return_value=None,
    )

    # on purpose transpose y_lf_mat here to check if this is wrong orientation is corrected
    mp2 = mocker.patch(
        'pqueens.models.simulation_model.SimulationModel.evaluate',
        return_value={"mean": y_lf_mat.T},
    )

    mp3 = mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_likelihood.BMFGaussianModel._evaluate_mf_likelihood',
        return_value=mf_log_likelihood_exp,
    )

    # pylint: enable=line-too-long

    mf_log_likelihood = default_mf_likelihood.evaluate(np.zeros(1))

    # assert statements
    mp1.assert_called_once()
    mp2.assert_called_once()
    mp3.assert_called_once()
    np.testing.assert_array_equal(y_lf_mat, mp3.call_args[0][0])
    np.testing.assert_array_equal(mf_log_likelihood, mf_log_likelihood_exp)


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
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_likelihood.BMFGaussianModel._calculate_distance_vector_and_var_y',
        return_value=(diff_mat, var_y_mat),
    )
    mp2 = mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_likelihood.BMFGaussianModel._calculate_likelihood_noise_var',
        return_value=None,
    )
    mp3 = mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_likelihood.BMFGaussianModel._log_likelihood_fun',
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


def test_calculate_distance_vector_and_var_y(default_mf_likelihood, mocker):
    """Test the calculation of the distance vector."""
    x_batch = np.array([[0, 0], [0, 1]])  # make points have distance 1
    y_lf_mat = np.array([[1, 1], [2, 2]])  # three dim output per point x in x_batch (row-wise)
    z_mat = np.array([[1, 2], [3, 4]])
    m_f_mat = np.array([[2, 2], [3, 3]])
    var_y_mat_exp = np.array([[7, 7], [7, 7]])
    coords_mat = default_mf_likelihood.coords_mat[: y_lf_mat.shape[0]]
    diff_mat_exp = np.array([[1, 0], [0, -1]])

    mp1 = mocker.patch(
        'pqueens.iterators.bmfia_iterator.BMFIAIterator._set_feature_strategy',
        return_value=z_mat,
    )
    mp2 = mocker.patch(
        'pqueens.interfaces.bmfia_interface.BmfiaInterface.evaluate',
        return_value=(m_f_mat, var_y_mat_exp),
    )

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


def test_calculate_likelihood_noise_var(default_mf_likelihood):
    """Test for (iterative) calculation of likelihood noise variance."""
    diff_mat = np.array([[1, 2], [2, 3]])

    default_mf_likelihood.likelihood_noise_type = "fixed"
    default_mf_likelihood._calculate_likelihood_noise_var(diff_mat)
    assert default_mf_likelihood.noise_var == default_mf_likelihood.fixed_likelihood_noise_value

    default_mf_likelihood.likelihood_noise_type = "jeffreys_prior"
    default_mf_likelihood._calculate_likelihood_noise_var(diff_mat)
    assert default_mf_likelihood.noise_var == 3.6


def test_calculate_rkhs_inner_product(default_mf_likelihood):
    """Test the calculation of the inner product and the different cases."""
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


def test_log_likelihood_fun(default_mf_likelihood):
    """Test the calculation of the actual multi-fidelity log-likelihood.

    At the moment we assume that we only consider the variance of
    vectorized output and not the entire covariance structure.
    """
    expected_fixed = np.array([[-70.2219]])
    expected_jeffreys = np.array([[-68.724]])
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


def test_initialize(default_mf_likelihood, mocker):
    """Test the initialization of the mf likelihood model."""
    coords_mat = np.array([[1, 2, 3], [2, 2, 2]])
    time_vec = np.linspace(1, 10, 3)
    y_obs = np.array([[5, 5, 5], [6, 6, 6]])

    default_mf_likelihood.coords_mat = coords_mat
    default_mf_likelihood.time_vec = time_vec
    default_mf_likelihood.y_obs = y_obs

    # pylint: disable=line-too-long
    mo_1 = mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_likelihood.print_bmfia_acceleration',
        return_value=None,
    )
    mo_2 = mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_likelihood.BMFGaussianModel._build_approximation',
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
        default_mf_likelihood.bmfia_subiterator.y_obs, y_obs, decimal=4
    )


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
        'pqueens.interfaces.bmfia_interface.BmfiaInterface.build_approximation',
        return_value=None,
    )
    mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_likelihood.qvis.bmfia_visualization_instance',
        return_value=mock_visualization,
    )
    mo_4 = mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_likelihood.qvis.bmfia_visualization_instance.plot',
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


def test_input_dim_red(default_mf_likelihood, mocker):
    """Test for the input dimensionality reduction routine."""
    # pylint: disable=line-too-long
    mo_1 = mocker.patch(
        "pqueens.models.likelihood_models.bayesian_mf_gaussian_likelihood.BMFGaussianModel.input_dim_red"
    )
    # pylint: enable=line-too-long
    default_mf_likelihood.input_dim_red()
    mo_1.assert_called_once()


def test_get_random_fields_and_truncated_basis(default_mf_likelihood):
    """Test the get-method for truncated random fields."""
    with pytest.raises(NotImplementedError):
        default_mf_likelihood.get_random_fields_and_truncated_basis()


def test_evaluate_forward_model(default_mf_likelihood, mock_model):
    """Test if forward model (lf model) is updated and evaluated correctly."""
    y_mat_expected = 1
    default_mf_likelihood.forward_model = mock_model
    y_mat = default_mf_likelihood.forward_model.evaluate(None)['mean']

    # actual tests / asserts
    np.testing.assert_array_almost_equal(y_mat, y_mat_expected, decimal=4)


def test_project_samples_on_truncated_basis(default_mf_likelihood):
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
