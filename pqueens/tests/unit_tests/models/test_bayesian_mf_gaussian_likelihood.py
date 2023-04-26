"""Unit tests for Bayesian multi-fidelity Gaussian likelihood function."""

import unittest.mock as mock

import numpy as np
import pytest
from mock import patch

import pqueens.models.likelihood_models.bayesian_mf_gaussian_likelihood as BMF
from pqueens.interfaces.bmfia_interface import BmfiaInterface
from pqueens.iterators.bmfia_iterator import BMFIAIterator
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
        "x1": {"type": "uniform", "lower_bound": -2, "upper_bound": 2},
        "x2": {"type": "uniform", "lower_bound": -2, "upper_bound": 2},
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
    coord_labels = ["x1", "x2"]
    time_vec = None
    coords_mat = np.array([[1, 0], [1, 0]])
    instantiate_probabilistic_mappings = BmfiaInterface._instantiate_per_coordinate
    evaluate_method = BmfiaInterface._evaluate_per_coordinate
    evaluate_and_gradient_method = BmfiaInterface._evaluate_and_gradient_per_coordinate
    update_mappings_method = BmfiaInterface._update_mappings_per_coordinate

    interface = BmfiaInterface(
        instantiate_probabilistic_mappings,
        num_processors_multi_processing,
        evaluate_method,
        evaluate_and_gradient_method,
        update_mappings_method,
    )
    interface.time_vec = time_vec
    interface.coords_mat = coords_mat
    interface.coord_labels = coord_labels
    return interface


@pytest.fixture()
def settings_probab_mapping(config, approximation_name):
    """Dummy settings for the probabilistic mapping for testing."""
    settings = config[approximation_name]
    return settings


@pytest.fixture()
def default_bmfia_iterator(global_settings):
    """Dummy iterator for testing."""
    global_settings = global_settings
    features_config = 'no_features'
    hf_model = 'dummy_hf_model'
    lf_model = 'dummy_lf_model'
    x_train = np.array([[1, 2], [3, 4]])
    Y_LF_train = np.array([[2], [3]])
    Y_HF_train = np.array([[2.2], [3.3]])
    Z_train = np.array([[4], [5]])
    coords_experimental_data = np.array([[1, 2], [3, 4]])
    time_vec = np.array([1, 3])
    y_obs = np.array([[2.1], [3.1]])
    x_cols = None
    num_features = None
    coord_cols = None

    with patch.object(BMFIAIterator, '_calculate_initial_x_train', lambda *args: x_train):
        iterator = BMFIAIterator(
            global_settings,
            features_config,
            hf_model,
            lf_model,
            initial_design={},
            X_cols=x_cols,
            num_features=num_features,
            coord_cols=coord_cols,
        )
    iterator.Y_LF_train = Y_LF_train
    iterator.Y_HF_train = Y_HF_train
    iterator.Z_train = Z_train
    iterator.coords_experimental_data = coords_experimental_data
    iterator.time_vec = time_vec
    iterator.y_obs_vec = y_obs

    return iterator


@pytest.fixture()
def default_mf_likelihood(
    dummy_model,
    parameters,
    default_interface,
    default_bmfia_iterator,
):
    """Default multi-fidelity Gaussian likelihood object."""
    forward_model = dummy_model
    coords_mat = np.array([[1, 2], [3, 4]])
    time_vec = np.array([1, 2, 3, 4])
    y_obs = np.array([[1, 2], [3, 4]])
    output_label = ['a', 'b']
    coord_labels = ['c', 'd']
    mf_interface = default_interface
    bmfia_subiterator = default_bmfia_iterator
    model_name = 'bmfia_model'
    noise_var = np.array([0.1])
    num_refinement_samples = 0
    likelihood_evals_for_refinement_lst = []
    dummy_normal_distr = "dummy"

    mf_likelihood = BMF.BMFGaussianModel(
        model_name,
        forward_model,
        coords_mat,
        time_vec,
        y_obs,
        output_label,
        coord_labels,
        mf_interface,
        bmfia_subiterator,
        dummy_normal_distr,
        noise_var,
        num_refinement_samples,
        likelihood_evals_for_refinement_lst,
    )

    return mf_likelihood


class InstanceMock:
    """Mock class."""

    def __init__(self, *args):
        """Initialize dummy class."""
        self.counter = 0

    def plot(self, *args, **kwargs):
        """Mock plot function."""
        self.counter += 1
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
def test_init(dummy_model, parameters, default_interface, default_bmfia_iterator):
    """Test the init of the multi-fidelity Gaussian likelihood function."""
    forward_model = dummy_model
    coords_mat = np.array([[1, 2], [3, 4]])
    time_vec = np.array([1, 2, 3, 4])
    y_obs = np.array([[1], [3]])
    output_label = ['a', 'b']
    coord_labels = ['c', 'd']
    mf_interface = default_interface
    bmfia_subiterator = default_bmfia_iterator
    model_name = 'bmfia_model'
    noise_var = None
    mean_field_normal = 'dummy'
    num_refinement_samples = 0
    likelihood_evals_for_refinement_lst = []

    model = BMF.BMFGaussianModel(
        model_name,
        forward_model,
        coords_mat,
        time_vec,
        y_obs,
        output_label,
        coord_labels,
        mf_interface,
        bmfia_subiterator,
        mean_field_normal,
        noise_var,
        num_refinement_samples,
        likelihood_evals_for_refinement_lst,
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
    assert model.bmfia_subiterator == bmfia_subiterator
    assert model.min_log_lik_mf is None
    assert model.normal_distribution == mean_field_normal
    assert model.noise_var is None
    assert model.likelihood_counter == 1
    assert model.num_refinement_samples == num_refinement_samples


def test_evaluate(default_mf_likelihood, mocker):
    """Compare return value with the expected value using a single point."""
    likelihood_output = np.array(np.array([7, 7]))
    y_lf_mat = np.array([[1, 2]])
    samples = np.array([[2, 3]])
    # pylint: disable=line-too-long
    # on purpose transpose y_lf_mat here to check if this is wrong orientation is corrected
    mp1 = mocker.patch(
        'pqueens.models.simulation_model.SimulationModel.evaluate',
        return_value={"mean": y_lf_mat.T},
    )

    mp2 = mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_likelihood.BMFGaussianModel.evaluate_from_output',
        return_value=likelihood_output,
    )

    # pylint: enable=line-too-long
    mf_log_likelihood = default_mf_likelihood.evaluate(samples)

    # assert statements
    mp1.assert_called_once()
    mp2.assert_called_once()
    np.testing.assert_array_equal(samples, mp1.call_args[0][0])
    np.testing.assert_array_equal(samples, mp2.call_args[0][0])
    np.testing.assert_array_equal(y_lf_mat, mp2.call_args[0][1])
    np.testing.assert_array_equal(mf_log_likelihood, likelihood_output)


def test_evaluate_from_output(default_mf_likelihood, mocker):
    """Compare return value with the expected value using a single point."""
    samples = np.array([[1, 2], [3, 4]])
    forward_model_output = np.array([[5], [6]])
    mf_log_likelihood_exp = np.array([[7], [9]])
    mp1 = mocker.patch(
        'BMF.BMFGaussianModel._evaluate_mf_likelihood',
        return_value=mf_log_likelihood_exp,
    )

    y_lf_mat = np.array([[1, 2]])

    # test without adaptivity
    mf_log_likelihood = default_mf_likelihood.evaluate_from_output(samples, forward_model_output)
    mp1.assert_called_once()
    mp1.assert_called_with(samples, forward_model_output)
    np.testing.assert_array_equal(mf_log_likelihood, mf_log_likelihood_exp)
    assert default_mf_likelihood.likelihood_counter == 2

    # test with adaptivity
    mocker.patch(
        'BMF.BMFGaussianModel._adaptivity_trigger',
        return_value=True,
    )
    mocker.patch('BMF.BMFGaussianModel._refine_mf_likelihood')
    with pytest.raises(NotImplementedError):
        mf_log_likelihood = default_mf_likelihood.evaluate_from_output(
            mf_log_likelihood_exp, y_lf_mat
        )


def test_evaluate_mf_likelihood(default_mf_likelihood, mocker):
    """Test the evaluation of the log multi-fidelity Gaussian likelihood."""
    # --- define some vectors and matrices -----
    y_lf_mat = np.array(
        [[1, 1, 1], [2, 2, 2]]
    )  # three dim output per point x in x_batch (row-wise)
    x_batch = np.array([[0, 0], [0, 1]])  # make points have distance 1
    diff_mat = np.array([[1, 1, 1], [2, 2, 2]])  # for simplicity we assume diff_mat equals
    var_y_mat = np.array([[1, 1, 1], [1, 1, 1]])
    z_mat = y_lf_mat
    m_f_mat = np.array([[1, 1], [1, 1]])

    # mock the normal distribution
    distribution_mock = mock.MagicMock()
    distribution_mock.update_variance.return_value = None
    distribution_mock.logpdf.return_value = np.array([[1.0]])
    default_mf_likelihood.normal_distribution = distribution_mock

    # pylint: disable=line-too-long
    mp1 = mocker.patch(
        'pqueens.iterators.bmfia_iterator.BMFIAIterator.set_feature_strategy',
        return_value=(z_mat),
    )
    mp2 = mocker.patch(
        'pqueens.interfaces.bmfia_interface.BmfiaInterface.evaluate',
        return_value=(m_f_mat, var_y_mat),
    )
    # pylint: enable=line-too-long

    log_lik_mf = default_mf_likelihood._evaluate_mf_likelihood(x_batch, y_lf_mat)

    # ------ assert and test statements ------------------------------------
    mp1.assert_called_once()
    np.testing.assert_array_equal(y_lf_mat, mp1.call_args[0][0])
    np.testing.assert_array_equal(x_batch, mp1.call_args[0][1])
    np.testing.assert_array_equal(
        default_mf_likelihood.coords_mat[: y_lf_mat.shape[0]], mp1.call_args[0][2]
    )

    # test evaluate method
    mp2.assert_called_once()
    np.testing.assert_array_equal(z_mat, mp2.call_args[0][0])

    # test covariance update and logpdf count
    assert distribution_mock.update_variance.call_count == diff_mat.shape[0]
    assert distribution_mock.logpdf.call_count == diff_mat.shape[0]

    # test logpdf output and input
    np.testing.assert_array_equal(log_lik_mf, np.array([[1], [1]]))


def test_evaluate_and_gradient(default_mf_likelihood):
    """Test the evaluate and gradient method."""
    # define inputs
    samples = np.array([[1, 2], [3, 4]])
    upstream_gradient_fun = lambda x: 2 * x

    # mock prepare downstream gradient fun
    dummy_fun = lambda x: x
    mp1 = mock.MagicMock(return_value=dummy_fun)

    # mock forward model evaluate and gradient fun
    sub_model_output = (np.array([[7], [8]]), np.array([[9, 9], [10, 10]]))
    mp2 = mock.MagicMock(return_value=sub_model_output)

    # mock evaluate from output fun
    log_lik_out = np.array([[11], [12]])
    mp3 = mock.MagicMock(return_value=log_lik_out)

    # pylint: disable=line-too-long
    with mock.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_likelihood.prepare_downstream_gradient_fun',
        mp1,
    ), mock.patch(
        'pqueens.models.simulation_model.SimulationModel.evaluate_and_gradient', mp2
    ), mock.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_likelihood.BMFGaussianModel.evaluate_from_output',
        mp3,
    ):
        log_likelihood, grad_objective_samples = default_mf_likelihood.evaluate_and_gradient(
            samples, upstream_gradient_fun
        )
    # pylint: enable=line-too-long

    # --- assert and test statements ------------------------------------
    # test prepare downstream gradient fun
    mp1.assert_called_once()
    mp1.assert_called_with(
        eval_output_fun=mp3,
        partial_grad_evaluate_fun=default_mf_likelihood.partial_grad_evaluate,
        upstream_gradient_fun=upstream_gradient_fun,
    )

    # test forward model evaluate and gradient fun
    mp2.assert_called_once()
    mp2.assert_called_with(samples, upstream_gradient_fun=dummy_fun)

    # test evaluate from output fun
    mp3.assert_called_once()
    mp3.assert_called_with(samples, sub_model_output[0])

    # test final method output
    np.testing.assert_array_equal(log_likelihood, log_lik_out)
    np.testing.assert_array_equal(grad_objective_samples, sub_model_output[1])


def test_partial_grad_evaluate(mocker, default_mf_likelihood):
    """Test the partial grad evaluate method."""
    # --- define some vectors and matrices -----
    forward_model_output = np.array(
        [[1, 1, 1], [2, 2, 2]]
    )  # three dim output per point x in x_batch (row-wise)
    forward_model_input = np.array([[0, 0], [0, 1]])  # make points have distance 1
    diff_mat = np.array([[1, 1, 1], [2, 2, 2]])  # for simplicity we assume diff_mat equals
    var_y_mat = np.array([[1, 1, 1], [1, 1, 1]])
    z_mat = forward_model_output
    m_f_mat = np.array([[1, 1], [1, 1]])
    grad_m_f_mat = np.array([[6, 7, 8], [9, 10, 11]])
    grad_var_y_mat = np.array([[12, 13, 14], [15, 16, 17]])

    # create mock attribute for mf_likelihood
    distribution_mock = mock.MagicMock()
    distribution_mock.update_variance.return_value = None
    distribution_mock.logpdf.return_value = np.array([[1]])
    default_mf_likelihood.normal_distribution = distribution_mock

    # pylint: disable=line-too-long
    mp1 = mocker.patch(
        'pqueens.iterators.bmfia_iterator.BMFIAIterator.set_feature_strategy',
        return_value=z_mat,
    )
    mp2 = mocker.patch(
        'pqueens.interfaces.bmfia_interface.BmfiaInterface.evaluate_and_gradient',
        return_value=(m_f_mat, var_y_mat, grad_m_f_mat, grad_var_y_mat),
    )

    mocker.patch(
        'pqueens.distributions.mean_field_normal.MeanFieldNormalDistribution.logpdf',
        return_value=0.1,
    )
    mp3 = mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_likelihood.BMFGaussianModel.grad_log_pdf_d_ylf',
        return_value=np.array([[0.2]]),
    )
    # pylint: enable=line-too-long
    grad_out = default_mf_likelihood.partial_grad_evaluate(
        forward_model_input, forward_model_output
    )

    # ------ assert and test statements ------------------------------------
    mp1.assert_called_once()
    np.testing.assert_array_equal(forward_model_output, mp1.call_args[0][0])
    np.testing.assert_array_equal(forward_model_input, mp1.call_args[0][1])
    np.testing.assert_array_equal(
        default_mf_likelihood.coords_mat[: forward_model_output.shape[0]], mp1.call_args[0][2]
    )

    # test evaluate method
    mp2.assert_called_once()
    np.testing.assert_array_equal(z_mat, mp2.call_args[0][0])

    # test covariance update and logpdf count
    assert distribution_mock.update_variance.call_count == diff_mat.shape[0]
    assert distribution_mock.logpdf.call_count == diff_mat.shape[0]

    # test grad logpdf features
    assert mp3.call_count == diff_mat.shape[0]

    # test logpdf output and input
    np.testing.assert_array_equal(grad_out, np.array([[0.2], [0.2]]))


def test_grad_log_pdf_d_ylf(default_mf_likelihood):
    """Test grad log pdf d ylf method."""
    m_f_vec = np.array([[1, 4]])
    grad_m_f = np.array([[3, 5]])
    grad_var_y = np.array([[7, 10]])

    d_log_lik_d_mf = np.array([[1], [2]])  # two samples, output is scalar
    d_log_lik_d_var = np.array([[3], [4]])  # two samples, output is scalar

    # create mock attribute for mf_likelihood
    distribution_mock = mock.MagicMock()
    distribution_mock.grad_logpdf.return_value = d_log_lik_d_mf
    distribution_mock.grad_logpdf_var.return_value = d_log_lik_d_var
    default_mf_likelihood.normal_distribution = distribution_mock

    # run the method
    d_log_lik_d_z = default_mf_likelihood.grad_log_pdf_d_ylf(m_f_vec, grad_m_f, grad_var_y)

    # --- assert and test statements ----------------------------------------
    distribution_mock.grad_logpdf.assert_called_once()
    distribution_mock.grad_logpdf.assert_called_with(m_f_vec)

    distribution_mock.grad_logpdf_var.assert_called_once()
    distribution_mock.grad_logpdf_var.assert_called_with(m_f_vec)

    expected_d_log_lik_d_z = np.array([[24, 50]])
    np.testing.assert_array_equal(d_log_lik_d_z, expected_d_log_lik_d_z)


def test_initialize_bmfia_iterator(default_bmfia_iterator, mocker):
    """Test the initialization of the mf likelihood model."""
    coords_mat = np.array([[1, 2, 3], [2, 2, 2]])
    time_vec = np.linspace(1, 10, 3)
    y_obs = np.array([[5, 5, 5], [6, 6, 6]])

    # pylint: disable=line-too-long
    mo_1 = mocker.patch(
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_likelihood.print_bmfia_acceleration',
        return_value=None,
    )

    # pylint: enable=line-too-long
    BMFGaussianModel.initialize_bmfia_iterator(coords_mat, time_vec, y_obs, default_bmfia_iterator)

    # actual tests / asserts
    mo_1.assert_called_once()
    np.testing.assert_array_almost_equal(
        default_bmfia_iterator.coords_experimental_data, coords_mat, decimal=4
    )
    np.testing.assert_array_almost_equal(default_bmfia_iterator.time_vec, time_vec, decimal=4)
    np.testing.assert_array_almost_equal(default_bmfia_iterator.y_obs, y_obs, decimal=4)


def test_build_approximation(default_bmfia_iterator, default_interface, config, mocker):
    """Test for the build stage of the probabilistic regression model."""
    z_train = np.array([[1, 1, 1], [2, 2, 2]])
    y_hf_train = np.array([[1, 1], [2, 2]])
    coord_labels = ['x', 'y', 'z']
    time_vec = default_bmfia_iterator.time_vec
    coords_mat = default_bmfia_iterator.coords_experimental_data
    approx_name = 'bmfia'

    # pylint: disable=line-too-long
    mo_1 = mocker.patch(
        'pqueens.iterators.bmfia_iterator.BMFIAIterator.core_run',
        return_value=(z_train, y_hf_train),
    )
    mo_2 = mocker.patch(
        'pqueens.interfaces.bmfia_interface.BmfiaInterface.build_approximation',
        return_value=None,
    )
    # pylint: enable=line-too-long

    BMF.BMFGaussianModel._build_approximation(
        default_bmfia_iterator,
        default_interface,
        config,
        approx_name,
        coord_labels,
        time_vec,
        coords_mat,
    )

    # actual asserts/tests
    mo_1.assert_called_once()
    mo_2.assert_called_once_with(
        z_train, y_hf_train, config, approx_name, coord_labels, time_vec, coords_mat
    )


def test_evaluate_forward_model(default_mf_likelihood, mock_model):
    """Test if forward model (lf model) is updated and evaluated correctly."""
    y_mat_expected = 1
    default_mf_likelihood.forward_model = mock_model
    y_mat = default_mf_likelihood.forward_model.evaluate(None)['mean']

    # actual tests / asserts
    np.testing.assert_array_almost_equal(y_mat, y_mat_expected, decimal=4)
