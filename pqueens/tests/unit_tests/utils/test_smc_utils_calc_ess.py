"""Test-module for calculation of effective sample size of smc_utils module.

@author: Sebastian Brandstaeter
"""
import numpy as np
import pytest

from pqueens.utils import smc_utils


@pytest.fixture(name="num_particles", scope='module', params=[1, 10])
def num_particles_fixture(request):
    """Return possible number of weights."""
    return request.param


def test_calc_ess_equal_weights(num_particles):
    """Test special case of resampled particles.

    For N resampled particles the weights are all equal (=1/N) and the
    ESS=N.
    """
    weights = np.array([1.0 / num_particles] * num_particles)
    ess_sol = num_particles
    ess = smc_utils.calc_ess(weights)

    assert np.isclose(ess, ess_sol)


def test_calc_ess():
    """Test ESS=0.5*N.

    The ess is a measure for the amount of potent particles. The higher
    the weight of a particle, the more potent it is. For particles with
    weights (close to) zero give to contribution to ess (**TODO_doc:**
    Please check this sentence). In case X percent of the particles have
    zero weight, the ess is N*(100%-X). E.g. half of the particles ->
    ESS = N/2.
    """
    num_particles = 10
    half_num_particles = int(0.5 * num_particles)
    weights = np.array([0.0] * half_num_particles + [1 / num_particles] * half_num_particles)
    ess_sol = half_num_particles
    ess = smc_utils.calc_ess(weights)

    assert np.isclose(ess, ess_sol)
