import numpy as np
import pytest

from pqueens.utils.stochastic_optimizer import StochasticOptimizer


@pytest.mark.integration_tests
def test_RMSprop_max(rmsprop_optimizer):
    varparams = 5 * np.ones(5).reshape(-1, 1)
    rmsprop_optimizer.current_variational_parameters = varparams
    rmsprop_optimizer.gradient = gradient
    result = None
    for r in rmsprop_optimizer:
        result = r

    iterations = rmsprop_optimizer.iteration
    assert iterations == 500
    assert np.mean(result - 0.5) < 0.05


@pytest.mark.integration_tests
def test_Adamax(adamax_optimizer):
    varparams = np.ones(5).reshape(-1, 1)
    adamax_optimizer.current_variational_parameters = varparams
    grad = lambda x: -gradient(x)
    adamax_optimizer.gradient = grad
    result = adamax_optimizer.run_optimization()
    iterations = adamax_optimizer.iteration
    assert iterations < 1000
    assert np.mean(result - 0.5) < 0.005


@pytest.mark.integration_tests
def test_Adam(adam_optimizer):
    varparams = np.ones(5).reshape(-1, 1)
    adam_optimizer.current_variational_parameters = varparams
    adam_optimizer.gradient = gradient
    result = adam_optimizer.run_optimization()
    iterations = adam_optimizer.iteration
    assert iterations < 1000
    assert np.mean(result - 0.5) < 0.005


@pytest.fixture()
def adam_optimizer():
    opt_config = {
        "dummy_section": {
            "stochastic_optimizer": "Adam",
            "learning_rate": 1e-2,
            "optimization_type": "max",
            "rel_L1_change_threshold": 1e-4,
            "rel_L2_change_threshold": 1e-6,
            "max_iter": 1000,
        }
    }
    optimizer = StochasticOptimizer.from_config_create_optimizer(
        opt_config, section_name="dummy_section"
    )
    return optimizer


@pytest.fixture()
def adamax_optimizer():
    opt_config = {
        "dummy_section": {
            "stochastic_optimizer": "Adamax",
            "learning_rate": 1e-2,
            "optimization_type": "min",
            "rel_L1_change_threshold": 1e-4,
            "rel_L2_change_threshold": 1e-6,
            "max_iter": 1000,
        }
    }
    optimizer = StochasticOptimizer.from_config_create_optimizer(
        opt_config, section_name="dummy_section"
    )
    return optimizer


@pytest.fixture()
def rmsprop_optimizer():
    opt_config = {
        "dummy_section": {
            "stochastic_optimizer": "RMSprop",
            "learning_rate": 5e-2,
            "optimization_type": "max",
            "rel_L1_change_threshold": -1,
            "rel_L2_change_threshold": -1,
            "max_iter": 500,
        }
    }
    optimizer = StochasticOptimizer.from_config_create_optimizer(
        opt_config, section_name="dummy_section"
    )
    return optimizer


def gradient(x):
    return -2 * (x - 0.5).reshape(-1, 1)
