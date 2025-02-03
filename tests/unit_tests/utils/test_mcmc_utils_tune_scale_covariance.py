#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Test-module for *tune_scale_covariance* of *mcmc_utils* module.

@author: Sebastian Brandstaeter
"""

import numpy as np
import pytest

from queens.utils.mcmc_utils import tune_scale_covariance


@pytest.fixture(
    name="accept_rate_and_scale_covariance",
    scope="module",
    params=[
        (1e-4, 0.1),
        (1e-2, 0.5),
        (1e-1, 0.9),
        (4e-1, 1.0),
        (6e-1, 1.1),
        (8e-1, 2.0),
        (9.9e-1, 10.0),
    ],
)
def fixture_accept_rate_and_scale_covariance(request):
    """Return a set of valid acceptance rate and adjusted scale.

    Given that the current scale is 1.0.
    """
    return request.param


def test_tune_scale_covariance(accept_rate_and_scale_covariance):
    """Test the tuning of scale covariance in MCMC methods."""
    accept_rate = accept_rate_and_scale_covariance[0]
    expected_scale = accept_rate_and_scale_covariance[1]
    current_scale = 1.0
    assert tune_scale_covariance(current_scale, accept_rate) == expected_scale


def test_tune_scale_covariance_multiple_chains():
    """Test scale tuning for MCMC methods with multiple chains.

    Test the tuning of proposal covariance in MCMC methods with multiple
    chains.

    We assume here to have 7 parallel chains, that correspond to all
    possible tuning factors.
    """
    accept_rate = np.array([[1e-4, 1e-2, 1e-1, 4e-1, 6e-1, 8e-1, 9.9e-1]]).T
    expected_scale = np.array([[0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0]]).T

    current_scale = np.ones((7, 1))
    assert np.allclose(tune_scale_covariance(current_scale, accept_rate), expected_scale)
