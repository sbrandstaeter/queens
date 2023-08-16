"""Integration test for Gaussian Neural Network regression model."""

import numpy as np
import pytest

from pqueens.models.surrogate_models.gaussian_neural_network import GaussianNeuralNetworkModel
from pqueens.tests.integration_tests.example_simulator_functions.park91a import park91a_hifi
from pqueens.tests.integration_tests.example_simulator_functions.sinus import (
    gradient_sinus_test_fun,
    sinus_test_fun,
)


@pytest.fixture()
def my_model():
    """Configuration for gaussian nn model."""
    model = GaussianNeuralNetworkModel(
        activation_per_hidden_layer_lst=["elu", "elu", "elu", "elu"],
        nodes_per_hidden_layer_lst=[20, 20, 20, 20],
        adams_training_rate=0.001,
        batch_size=50,
        num_epochs=3000,
        optimizer_seed=42,
        data_scaling="standard_scaler",
        nugget_std=1.0e-02,
        verbosity_on=False,
    )
    return model


def test_gaussian_nn_one_dim(my_model):
    """Test one dimensional gaussian nn."""
    n_train = 25
    x_train = np.linspace(-5, 5, n_train).reshape(-1, 1)
    y_train = sinus_test_fun(x_train)

    my_model.setup(x_train, y_train)
    my_model.train()

    # evaluate the testing/benchmark function at testing inputs
    x_test = np.linspace(-5, 5, 200).reshape(-1, 1)
    mean_ref, gradient_mean_ref = gradient_sinus_test_fun(x_test)
    var_ref = np.zeros(mean_ref.shape)

    # --- get the mean and variance of the model (no gradient call here) ---
    output = my_model.predict(x_test)
    mean = output['mean']
    variance = output['variance']

    np.testing.assert_array_almost_equal(mean, mean_ref, decimal=2)
    np.testing.assert_array_almost_equal(variance, var_ref, decimal=2)

    # -- now call the gradient function of the model---
    output = my_model.predict(x_test, gradient_bool=True)
    mean = output['mean']
    variance = output['variance']
    gradient_mean = output['grad_mean']
    gradient_variance = output['grad_var']
    gradient_variance_ref = np.zeros(gradient_mean_ref.shape)

    np.testing.assert_array_almost_equal(mean, mean_ref, decimal=1)
    np.testing.assert_array_almost_equal(variance, var_ref, decimal=2)

    np.testing.assert_array_almost_equal(gradient_mean, gradient_mean_ref, decimal=1)
    np.testing.assert_array_almost_equal(gradient_variance, gradient_variance_ref, decimal=2)


def test_gaussian_nn_two_dim(my_model):
    """Test two dimensional gaussian nn."""
    n_train = 7
    x_3, x_4 = 0.5, 0.5
    x_1 = np.linspace(0.001, 0.999, n_train)
    x_2 = np.linspace(0.001, 0.999, n_train)
    xx_1, xx_2 = np.meshgrid(x_1, x_2)
    x_train = np.vstack((xx_1.flatten(), xx_2.flatten())).T

    # evaluate the testing/benchmark function at training inputs, train model
    y_train = park91a_hifi(x_train[:, 0], x_train[:, 1], x_3, x_4, gradient_bool=False)
    y_train = y_train.reshape(-1, 1)
    my_model.setup(x_train, y_train)
    my_model.train()

    # evaluate the testing/benchmark function at testing inputs
    n_test = 25
    x_3, x_4 = 0.5, 0.5
    x_1 = np.linspace(0.001, 0.999, n_test)
    x_2 = np.linspace(0.001, 0.999, n_test)
    xx_1, xx_2 = np.meshgrid(x_1, x_2)
    x_test = np.vstack((xx_1.flatten(), xx_2.flatten())).T

    mean_ref, gradient_mean_ref = park91a_hifi(
        x_test[:, 0], x_test[:, 1], x_3, x_4, gradient_bool=True
    )
    mean_ref = mean_ref.reshape(-1, 1)
    gradient_mean_ref = np.array(gradient_mean_ref).T
    var_ref = np.zeros(mean_ref.shape)

    # --- get the mean and variance of the model (no gradient call here) ---
    output = my_model.predict(x_test)
    mean = output['mean']
    variance = output['variance']

    np.testing.assert_array_almost_equal(mean, mean_ref, decimal=2)
    np.testing.assert_array_almost_equal(variance, var_ref, decimal=2)

    # -- now call the gradient function of the model---
    output = my_model.predict(x_test, gradient_bool=True)
    mean = output['mean']
    variance = output['variance']
    gradient_mean = output['grad_mean']
    gradient_variance = output['grad_var']
    gradient_variance_ref = np.zeros(gradient_mean_ref.shape)

    np.testing.assert_array_almost_equal(mean, mean_ref, decimal=2)
    np.testing.assert_array_almost_equal(variance, var_ref, decimal=2)

    np.testing.assert_array_almost_equal(gradient_mean, gradient_mean_ref, decimal=1)
    np.testing.assert_array_almost_equal(gradient_variance, gradient_variance_ref, decimal=2)
