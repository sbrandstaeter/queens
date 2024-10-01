"""Unit tests for the elementary effects iterator."""

import numpy as np
import pytest

from queens.iterators.elementary_effects_iterator import ElementaryEffectsIterator


@pytest.fixture(name="default_elementary_effects_iterator")
def fixture_default_elementary_effects_iterator(
    global_settings, default_simulation_model, default_parameters_uniform_3d
):
    """Default elementary effects iterator."""
    default_simulation_model.driver.parameters = default_parameters_uniform_3d

    my_iterator = ElementaryEffectsIterator(
        model=default_simulation_model,
        parameters=default_parameters_uniform_3d,
        global_settings=global_settings,
        num_trajectories=20,
        local_optimization=True,
        num_optimal_trajectories=4,
        number_of_levels=4,
        seed=42,
        confidence_level=0.95,
        num_bootstrap_samples=1000,
        result_description={},
    )
    return my_iterator


def test_correct_sampling(default_elementary_effects_iterator):
    """Test if sampling works correctly."""
    default_elementary_effects_iterator.pre_run()

    ref_vals = np.array(
        [
            [-1.04719755, 3.14159265, 3.14159265],
            [3.14159265, 3.14159265, 3.14159265],
            [3.14159265, 3.14159265, -1.04719755],
            [3.14159265, -1.04719755, -1.04719755],
            [-3.14159265, -1.04719755, -3.14159265],
            [-3.14159265, 3.14159265, -3.14159265],
            [-3.14159265, 3.14159265, 1.04719755],
            [1.04719755, 3.14159265, 1.04719755],
            [-3.14159265, -3.14159265, 1.04719755],
            [-3.14159265, -3.14159265, -3.14159265],
            [-3.14159265, 1.04719755, -3.14159265],
            [1.04719755, 1.04719755, -3.14159265],
            [3.14159265, 1.04719755, 3.14159265],
            [3.14159265, -3.14159265, 3.14159265],
            [-1.04719755, -3.14159265, 3.14159265],
            [-1.04719755, -3.14159265, -1.04719755],
        ]
    )

    np.testing.assert_allclose(default_elementary_effects_iterator.samples, ref_vals, 1e-07, 1e-07)


def test_correct_sensitivity_indices(default_elementary_effects_iterator):
    """Test correct results."""
    default_elementary_effects_iterator.pre_run()
    default_elementary_effects_iterator.core_run()
    si = default_elementary_effects_iterator.si

    ref_mu = np.array([10.82845216, 0.0, -3.12439805])
    ref_mu_star = np.array([10.82845216, 7.87500000, 3.12439805])
    ref_mu_star_conf = np.array([5.49677290, 0.0, 5.26474752])
    ref_sigma = np.array([6.24879610, 9.09326673, 6.24879610])

    np.testing.assert_allclose(si["mu"], ref_mu, 1e-07, 1e-07)
    np.testing.assert_allclose(si["mu_star"], ref_mu_star, 1e-07, 1e-07)
    np.testing.assert_allclose(si["mu_star_conf"], ref_mu_star_conf, 1e-07, 1e-07)
    np.testing.assert_allclose(si["sigma"], ref_sigma, 1e-07, 1e-07)
