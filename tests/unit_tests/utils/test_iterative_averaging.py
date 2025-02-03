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
"""Test iterative averaging utils."""

import numpy as np
import pytest

from queens.utils.iterative_averaging_utils import (
    ExponentialAveraging,
    MovingAveraging,
    PolyakAveraging,
    l1_norm,
    l2_norm,
    relative_change,
)


def test_l1_norm():
    """Test L1 norm."""
    x = 2 * np.ones(10)
    norm_l1 = l1_norm(x)
    norm_l1_avg = l1_norm(x, True)
    np.testing.assert_almost_equal(norm_l1, 20)
    np.testing.assert_almost_equal(norm_l1_avg, 2)


def test_l2_norm():
    """Test L2 norm."""
    x = 2 * np.ones(10)
    norm_l2 = l2_norm(x)
    norm_l2_avg = l2_norm(x, True)
    np.testing.assert_almost_equal(norm_l2, 2 * np.sqrt(10))
    np.testing.assert_almost_equal(norm_l2_avg, 2)


def test_relative_change():
    """Test relative change."""
    old = np.ones(10)
    new = np.ones(10) * 2
    rel_change = relative_change(old, new, l1_norm)
    np.testing.assert_almost_equal(rel_change, 1)


def test_polyak_averaging(type_of_averaging_quantity):
    """Test Polyak averaging."""
    polyak = PolyakAveraging()
    for j in range(10):
        polyak.update_average(type_of_averaging_quantity * j)
    np.testing.assert_equal(
        polyak.current_average, type_of_averaging_quantity * np.mean(np.arange(10))
    )


def test_moving_averaging(type_of_averaging_quantity):
    """Test moving averaging."""
    moving = MovingAveraging(5)
    for j in range(10):
        moving.update_average(type_of_averaging_quantity * j)
    np.testing.assert_equal(
        moving.current_average, type_of_averaging_quantity * np.mean(np.arange(0, 10)[-5:])
    )


def test_exponential_averaging(type_of_averaging_quantity):
    """Test exponential averaging."""
    alpha = 0.25
    exponential_avg = ExponentialAveraging(alpha)
    for j in range(10):
        exponential_avg.update_average(type_of_averaging_quantity * j)
    # For this special case there is a analytical solution
    ref = np.sum((1 - alpha) * np.arange(1, 10) * alpha ** np.arange(0, 9)[::-1])
    np.testing.assert_equal(exponential_avg.current_average, type_of_averaging_quantity * ref)


@pytest.fixture(
    name="type_of_averaging_quantity",
    scope="module",
    params=[1, np.arange(5), np.arange(5).reshape(-1, 1)],
)
def fixture_type_of_averaging_quantity(request):
    """Different objects for which to test averaging."""
    return request.param
