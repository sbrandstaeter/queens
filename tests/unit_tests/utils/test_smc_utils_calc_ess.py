"""Test-module for calculation of effective sample size of smc_utils module.

@author: Sebastian Brandstaeter
"""

import numpy as np
import pytest

from queens.utils import smc_utils


@pytest.fixture(name="num_particles", scope="module", params=[1, 10])
def fixture_num_particles(request):
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

    The effective sample size (ESS) is a measure for the amount of
    potent particles. The higher the weight of a particle, the more
    potent it is. Particles with weights close to zero contribute little
    to the ESS. If X percent of the N particles have zero weight and all
    remaining particles have the same non-zero weight, the ESS is
    N*(100%-X). For example, if half of the particles have zero weight
    and the other half has weight 1/N, ESS = N/2.
    """
    num_particles = 10
    half_num_particles = int(0.5 * num_particles)
    weights = np.array([0.0] * half_num_particles + [1 / num_particles] * half_num_particles)
    ess_sol = half_num_particles
    ess = smc_utils.calc_ess(weights)

    assert np.isclose(ess, ess_sol)
