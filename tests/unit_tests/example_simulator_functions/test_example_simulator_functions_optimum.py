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
"""Test module for example simulator functions optimum."""

import numpy as np

from example_simulator_functions.branin78 import branin78_hifi
from example_simulator_functions.forrester import forrester


def test_branin78_hifi_global_minima():
    """Test the three global minima of the Branin function."""
    expected = 5.0 / (4.0 * np.pi)

    result = branin78_hifi(-np.pi, 12.275)
    np.testing.assert_allclose(result, expected, rtol=1e-10)

    result = branin78_hifi(np.pi, 2.275)
    np.testing.assert_allclose(result, expected, rtol=1e-10)

    result = branin78_hifi(3.0 * np.pi, 2.475)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_forrester__global_minima():
    """Test the one global minimum of the Forrester function."""
    expected = -6.020740055767083

    result = forrester(0.757248757841856)
    np.testing.assert_allclose(result, expected, rtol=1e-10)
