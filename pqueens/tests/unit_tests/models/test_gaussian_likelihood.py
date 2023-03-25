"""Unit tests for gaussian likelihood model."""
from collections import namedtuple

import numpy as np
import pytest

from pqueens.distributions import from_config_create_distribution
from pqueens.models.likelihood_models.gaussian_likelihood import GaussianLikelihood

'''
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
                out.append(2 * y)
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
        "pqueens.models.likelihood_models.gaussian_likelihood.LikelihoodModel.get_base_attributes_from_config",
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
        "pqueens.models.likelihood_models.gaussian_likelihood.GaussianLikelihood.update_covariance"
    )
    response = my_lik_model.evaluate(samples)
    assert m1.called_once_with(3.0)


def test_evaluate_from_output(my_lik_model):
    """Test for the evaluate from output method."""
    # test working evaluation with forward model output
    response = my_lik_model.evaluate_from_output(3.0)
    assert response == 9


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


def test_evaluate_and_gradient(my_lik_model):
    """Test for the evaluate and gradient method."""
    samples = np.array([[1.0, 2.0], [2.0, 3.0]])

    def upstream_gradient_fun(x_batch, l_vec):
        """Return a dummy upstream gradient fun."""
        up_grad = []
        for l in l_vec:
            up_grad.append(l * l)

        up_grad = np.array(up_grad).reshape(-1, 1)
        return up_grad

    # Design a fake forward model class/object for this test
    class FakeModel:
        """A fake forward model class for testing."""

        def evaluate(self, samples):
            """Evaluate the model."""
            out = []
            for sample in samples:
                out.append(sample**2)
            out = np.array(out).reshape(-1, 1)
            return out

        def evaluate_and_gradient(self, samples, upstream_gradient_fun=None):
            """A fake eval and gradient method for testing."""
            model_output = self.evaluate(samples)
            grad_upstream = upstream_gradient_fun(samples, model_output)
            return model_output, grad_upstream

    # test evaluate and gradient without obj fun gradient
    my_lik_model.forward_model = FakeModel()
    log_likelihood, grad_log_likelihood = my_lik_model.evaluate_and_gradient(samples)
    expected_log_lik = np.array([[1.0, 16, 16, 81]]).T
    expected_grad_log_lik = np.array([[2, 8, 8, 18]]).T

    np.testing.assert_array_equal(log_likelihood, expected_log_lik)
    np.testing.assert_array_equal(grad_log_likelihood, expected_grad_log_lik)

    # test evaluate and gradient with obj fun gradient
    log_likelihood, upstream_gradient = my_lik_model.evaluate_and_gradient(
        samples, upstream_gradient_fun=upstream_gradient_fun
    )

    # new grad is different from before!
    expected_grad_objective = np.array([[2.00000e00, 2.04800e03, 2.04800e03, 1.18098e05]])

    np.testing.assert_array_equal(log_likelihood, expected_log_lik)
    np.testing.assert_array_equal(upstream_gradient, expected_grad_objective)


def test_partial_grad_evaluate(my_lik_model):
    """Test partial gradient of evaluation method."""
    samples = np.array([[1.0, 2.0], [2.0, 3.0]])
    forward_model_output = np.array([[1], [2]])
    grad_out = my_lik_model.partial_grad_evaluate(samples, forward_model_output)
    expected_grad_out = np.array([[2], [4]])

    np.testing.assert_array_equal(grad_out, expected_grad_out)
'''
