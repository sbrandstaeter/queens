"""Test stochastic optimizers."""
import numpy as np
import pytest

from pqueens.utils.stochastic_optimizer import from_config_create_optimizer


def test_RMSprop_max(rmsprop_optimizer):
    """Test RMSprop."""
    varparams = 5 * np.ones(5).reshape(-1, 1)
    rmsprop_optimizer.current_variational_parameters = varparams
    rmsprop_optimizer.set_gradient_function(gradient)
    result = None
    for r in rmsprop_optimizer:
        result = r

    iterations = rmsprop_optimizer.iteration
    assert iterations == 500
    assert np.mean(result - 0.5) < 0.05


def test_Adamax(adamax_optimizer):
    """Test Adamax."""
    varparams = np.ones(5).reshape(-1, 1)
    adamax_optimizer.current_variational_parameters = varparams
    adamax_optimizer.set_gradient_function(lambda x: -gradient(x))
    result = adamax_optimizer.run_optimization()
    iterations = adamax_optimizer.iteration
    assert iterations < 1000
    assert np.mean(result - 0.5) < 0.005


def test_Adam(adam_optimizer):
    """Test Adam."""
    varparams = np.ones(5).reshape(-1, 1)
    adam_optimizer.current_variational_parameters = varparams
    adam_optimizer.set_gradient_function(gradient)
    result = adam_optimizer.run_optimization()
    iterations = adam_optimizer.iteration
    assert iterations < 1000
    assert np.mean(result - 0.5) < 0.005


@pytest.fixture()
def adam_optimizer():
    """Adam optimizer."""
    opt_config = {
        "optimizer": {
            "type": "adam",
            "learning_rate": 1e-2,
            "optimization_type": "max",
            "rel_l1_change_threshold": 1e-4,
            "rel_l2_change_threshold": 1e-6,
            "max_iteration": 1000,
        }
    }
    optimizer = from_config_create_optimizer(opt_config, optimizer_name="optimizer")
    return optimizer


@pytest.fixture()
def adamax_optimizer():
    """Adamax optimizer."""
    opt_config = {
        "optimizer": {
            "type": "adamax",
            "learning_rate": 1e-2,
            "optimization_type": "min",
            "rel_l1_change_threshold": 1e-4,
            "rel_l2_change_threshold": 1e-6,
            "max_iteration": 1000,
        }
    }
    optimizer = from_config_create_optimizer(opt_config, optimizer_name="optimizer")
    return optimizer


@pytest.fixture()
def rmsprop_optimizer():
    """Rmsprop optimzer."""
    opt_config = {
        "optimizer": {
            "type": "rms_prop",
            "learning_rate": 5e-2,
            "optimization_type": "max",
            "rel_l1_change_threshold": -1,
            "rel_l2_change_threshold": -1,
            "max_iteration": 500,
        }
    }
    optimizer = from_config_create_optimizer(opt_config, optimizer_name="optimizer")
    return optimizer


def gradient(x):
    """Gradient function."""
    return -2 * (x - 0.5).reshape(-1, 1)
