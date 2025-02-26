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
"""Test-module for numpy utils functions."""

import numpy as np
import pytest

from queens.utils.numpy import at_least_2d, at_least_3d


@pytest.fixture(name="arr_0d", scope="module")
def fixture_arr_0d():
    """Return possible number of weights."""
    return np.random.rand(1).squeeze()


@pytest.fixture(name="arr_1d", scope="module")
def fixture_arr_1d():
    """Return 1D array."""
    return np.random.rand(3)


@pytest.fixture(name="arr_2d", scope="module")
def fixture_arr_2d():
    """Return 2D array."""
    return np.random.rand(4, 2)


@pytest.fixture(name="arr_3d", scope="module")
def fixture_arr_3d():
    """Return 3D array."""
    return np.random.rand(3, 2, 5)


def test_at_least_2d(arr_0d, arr_1d, arr_2d):
    """Test numpy utils function *at_least_2d*."""
    np.testing.assert_equal(at_least_2d(arr_0d).shape, (1, 1))
    np.testing.assert_equal(at_least_2d(arr_1d).shape, (arr_1d.shape[0], 1))
    np.testing.assert_equal(at_least_2d(arr_2d).shape, arr_2d.shape)


def test_at_least_3d(arr_0d, arr_1d, arr_2d, arr_3d):
    """Test numpy utils function *at_least_3d*."""
    np.testing.assert_equal(at_least_3d(arr_0d).shape, (1, 1, 1))
    np.testing.assert_equal(at_least_3d(arr_1d).shape, (arr_1d.shape[0], 1, 1))
    np.testing.assert_equal(at_least_3d(arr_2d).shape, (arr_2d.shape[0], arr_2d.shape[1], 1))
    np.testing.assert_equal(at_least_3d(arr_3d).shape, arr_3d.shape)
