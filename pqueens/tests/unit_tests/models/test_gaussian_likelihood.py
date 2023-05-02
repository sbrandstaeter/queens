"""Unit tests for gaussian likelihood model."""
from collections import namedtuple

import numpy as np
import pytest
from mock import Mock

from pqueens.distributions import from_config_create_distribution
from pqueens.models.likelihood_models.gaussian_likelihood import GaussianLikelihood


# ---------------- some fixtures ---------------------------------- #
@pytest.fixture()
def dummy_config():
    """A dummy config dictionary."""
    config = {
        "my_lik_model": {
            "nugget_noise_variance": 1e-6,
            "forward_model_name": "my_forward_model",
            "noise_type": "fixed_variance",
            "noise_value": 1e-3,
            "noise_var_iterative_averaging": None,
            "normal_distribution": "dummy_distr",
            "coords_mat": np.array([[1.0]]),
            "time_vec": np.array([1.0, 2.0]),
            "y_obs": np.array([[3.0]]),
            "output_label": ["out"],
            "coord_labels": ["c1", "c2"],
        },
        "my_forward_model": "dummy_model",
    }
    return config


@pytest.fixture()
def my_lik_model():
    """Fixture for dummy likelihood model."""
    forward_model_dummy = namedtuple("forward_model", ["evaluate"])

    class FakeDistr:
        """A fake distribution class."""

        def logpdf(self, x):
            """A fake logpdf method."""
            return x**2

        def evaluate(self, x):
            """A fake eval method."""
            return x**2

        def update_covariance(self, x):
            """A fake eval method."""
            self.cov = x

        def grad_logpdf(self, y_vec):
            """A fake grad logpdf fun."""
            out = []
            for y in y_vec:
                out.append(-2 * np.linalg.norm(y))
            return np.array(out).reshape(-1, 1)

    distr_dummy = FakeDistr()

    model_name = "my_lik_model"
    nugget_noise_variance = 0.1
    forward_model = forward_model_dummy(lambda x: {"mean": x + 1})
    noise_type = "static"
    noise_var_iterative_averaging = None
    normal_distribution = distr_dummy
    coords_mat = np.array([[1.0]])
    time_vec = np.array([1.0, 2.0])
    y_obs = np.array([[3.0]])
    output_label = ["out"]
    coord_labels = ["c1", "c2"]

    gauss_lik_obj = GaussianLikelihood(
        model_name,
        nugget_noise_variance,
        forward_model,
        noise_type,
        noise_var_iterative_averaging,
        normal_distribution,
        coords_mat,
        time_vec,
        y_obs,
        output_label,
        coord_labels,
    )
    return gauss_lik_obj


# ----------------- actual unit tests ------------------------------#
def test_init():
    """Test for the init method."""
    model_name = "my_lik_model"
    nugget_noise_variance = 0.1
    forward_model = "my_forward_model"
    noise_type = "static"
    noise_var_iterative_averaging = None
    normal_distribution = "dummy_distr"
    coords_mat = np.array([[1.0]])
    time_vec = np.array([1.0, 2.0])
    y_obs = np.array([[3.0]])
    output_label = ["out"]
    coord_labels = ["c1", "c2"]

    gauss_lik_obj = GaussianLikelihood(
        model_name,
        nugget_noise_variance,
        forward_model,
        noise_type,
        noise_var_iterative_averaging,
        normal_distribution,
        coords_mat,
        time_vec,
        y_obs,
        output_label,
        coord_labels,
    )
    assert gauss_lik_obj.name == model_name
    assert gauss_lik_obj.nugget_noise_variance == nugget_noise_variance
    assert gauss_lik_obj.forward_model == forward_model
    assert gauss_lik_obj.noise_type == noise_type
    assert gauss_lik_obj.noise_var_iterative_averaging == noise_var_iterative_averaging
    assert gauss_lik_obj.normal_distribution == normal_distribution
    assert gauss_lik_obj.coords_mat == coords_mat
    np.testing.assert_array_equal(gauss_lik_obj.time_vec, time_vec)
    assert gauss_lik_obj.y_obs == y_obs
    assert gauss_lik_obj.output_label == output_label
    assert gauss_lik_obj.coord_labels == coord_labels
    assert gauss_lik_obj.__class__.__name__ == "GaussianLikelihood"


def test_fcc(dummy_config, mocker):
    """Test for the fcc method."""
    model_name = "my_lik_model"
    forward_model = "my_forward_model"
    coords_mat = np.array([[1.0, 1.0]])
    time_vec = np.array([[2.0, 2.0]])
    y_obs = np.array([3.0])
    output_label = ["y_obs"]
    coord_labels = ["c1", "c2"]
    m1 = mocker.patch(
        "pqueens.models.likelihood_models.gaussian_likelihood."
        "LikelihoodModel.get_base_attributes_from_config",
        return_value=(
            forward_model,
            coords_mat,
            time_vec,
            y_obs,
            output_label,
            coord_labels,
        ),
    )
    # create the normal distribution of the Gaussian likelihood model for testing
    covariance = dummy_config[model_name]["noise_value"] * np.eye(y_obs.size)
    distribution_options = {"type": "normal", "mean": y_obs, "covariance": covariance}
    normal_distribution = from_config_create_distribution(distribution_options)

    # test valid configuration
    gauss_lik_obj = GaussianLikelihood.from_config_create_model(model_name, dummy_config)
    assert gauss_lik_obj.__class__.__name__ == "GaussianLikelihood"
    assert gauss_lik_obj.name == model_name
    assert gauss_lik_obj.nugget_noise_variance == dummy_config[model_name]["nugget_noise_variance"]
    assert gauss_lik_obj.forward_model == forward_model
    assert gauss_lik_obj.noise_type == dummy_config[model_name]["noise_type"]
    assert (
        gauss_lik_obj.noise_var_iterative_averaging
        == dummy_config[model_name]["noise_var_iterative_averaging"]
    )
    assert gauss_lik_obj.normal_distribution.mean == normal_distribution.mean
    assert gauss_lik_obj.normal_distribution.covariance == normal_distribution.covariance
    np.testing.assert_equal(gauss_lik_obj.coords_mat, coords_mat)
    np.testing.assert_array_equal(gauss_lik_obj.time_vec, time_vec)
    assert gauss_lik_obj.y_obs == y_obs
    assert gauss_lik_obj.output_label == output_label
    assert gauss_lik_obj.coord_labels == coord_labels
    assert m1.called_once()


def test_evaluate(mocker, my_lik_model):
    """Test for the evaluate method."""
    samples = np.array([[1.0]])
    # test working evaluation
    response = my_lik_model.evaluate(samples)
    assert response == 4

    # test update of covariance for MAP
    my_lik_model.noise_type = "MAP_abc"
    m1 = mocker.patch(
        "pqueens.models.likelihood_models.gaussian_likelihood."
        "GaussianLikelihood.update_covariance"
    )
    response = my_lik_model.evaluate(samples)
    assert m1.called_once_with(3.0)


def test_update_covariance(my_lik_model):
    """Test for update of covariance."""
    y_model = np.array([[1.0, 7.0]])

    # test MAP jeffrey variance, no averaging
    my_lik_model.noise_type = "MAP_jeffrey_variance"
    my_lik_model.update_covariance(y_model)
    expected_cov = np.array([[2, 0], [0, 2]])
    # we duck-typed the normal distribution obj and write its argument into an attribute
    np.testing.assert_array_equal(my_lik_model.normal_distribution.cov, expected_cov)

    # test MAP jeffery variance vector, no averaging
    my_lik_model.noise_type = "MAP_jeffrey_variance_vector"
    my_lik_model.update_covariance(y_model)
    expected_cov = np.array([[0.8, 0], [0, 3.2]])
    np.testing.assert_array_equal(my_lik_model.normal_distribution.cov, expected_cov)

    # test other MAP case, no averaging
    my_lik_model.noise_type = "MAP_abcdefg"
    my_lik_model.update_covariance(y_model)
    expected_cov = np.array([[0.8, -1.6], [-1.6, 3.2]])
    np.testing.assert_array_equal(my_lik_model.normal_distribution.cov, expected_cov)

    # test other MAP case with averaging
    class DummyClass:
        """Dummy class for testing."""

        def __init__(self):
            """Dummy init method."""
            self.cov = None

        def update_average(self, cov):
            """Dummy method for testing."""
            self.cov = cov
            return cov

    dummy_averaging_obj = DummyClass()
    my_lik_model.noise_var_iterative_averaging = dummy_averaging_obj
    my_lik_model.update_covariance(y_model)
    expected_cov = np.array([[0.8, -1.6], [-1.6, 3.2]])

    # actual tests
    np.testing.assert_array_equal(my_lik_model.normal_distribution.cov, expected_cov)
    # also check if method was actually called (we just write the method argument as an attribute
    # of the dummy obj here to make this check
    np.testing.assert_array_equal(my_lik_model.noise_var_iterative_averaging.cov, expected_cov)


def test_grad(my_lik_model):
    """Test grad method."""
    samples = np.array([[1.0, 2.0], [2.0, 3.0]])
    my_lik_model.evaluate(samples)
    my_lik_model.forward_model = Mock()
    my_lik_model.forward_model.grad = lambda _samples, _upstream: _samples + _upstream
    upstream = np.array([[6], [7]])
    grad = my_lik_model.grad(samples, upstream=upstream)
    expected_grad = np.array([[-42.2666153056, -41.2666153056], [-68.0000000000, -67.0000000000]])
    np.testing.assert_almost_equal(expected_grad, grad)
