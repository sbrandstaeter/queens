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
"""Test-module for finite difference based computation of Jacobian.

@author: Sebastian Brandstaeter
"""

import numpy as np
import pytest
from scipy.optimize import rosen
from scipy.optimize._numdiff import approx_derivative

from queens.utils.fd_jacobian import fd_jacobian, get_positions


@pytest.fixture(name="method", scope="module", params=["2-point", "3-point"])
def fixture_method(request):
    """All possible finite difference schemes."""
    return request.param


@pytest.fixture(name="rel_step", scope="module", params=[None, 0.1])
def fixture_rel_step(request):
    """Whether user specified a relative step size."""
    return request.param


@pytest.fixture(name="x0", scope="module")
def fixture_x0():
    """Position where Jacobian should be evaluated."""
    return np.array([-3.0, 4.0, 0.0])


@pytest.fixture(
    name="bounds",
    scope="module",
    params=[
        (-np.inf, np.inf),  # no bounds
        ([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]),  # inactive bounds
        ([-3.0, -10.0, -10.0], [10.0, 10.0, 10.0]),
        ([-10.0, 4.0, -10.0], [10.0, 10.0, 10.0]),
        ([-10.0, -10.0, 0.0], [10.0, 10.0, 10.0]),
        ([-10.0, -10.0, -10.0], [-3.0, 10.0, 10.0]),
        ([-10.0, -10.0, -10.0], [10.0, 4.0, 10.0]),
        ([-10.0, -10.0, -10.0], [10.0, 10.0, 0.0]),
    ],
)
def fixture_bounds(request):
    """Possible combination of bounds."""
    return request.param


def test_fd_jacobian(x0, method, rel_step, bounds):
    """Test reimplementation of Jacobian against the original.

    Cover all possible parameter combinations. Based on the Rosenbrock
    test function supplied by *scipy.optimize*.
    """
    # calculated all necessary inputs

    x_stencil_batch, dx, use_one_sided = get_positions(x0, method, rel_step, bounds)

    x_batch = np.vstack((np.atleast_2d(x0), x_stencil_batch))

    f_batch = np.array([[rosen(x)] for x in x_batch])

    f0 = f_batch[0]  # first entry corresponds to f(x0)
    f_perturbed = np.delete(f_batch, 0, 0)  # delete the first entry

    expected_jacobian = approx_derivative(
        rosen, x0, method, rel_step=rel_step, f0=None, bounds=bounds
    )
    actual_jacobian = fd_jacobian(f0, f_perturbed, dx, use_one_sided, method)

    np.testing.assert_allclose(np.squeeze(expected_jacobian), actual_jacobian)
