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
"""Numpy array utils."""

import numpy as np


def at_least_2d(arr):
    """View input array as array with at least two dimensions.

    Args:
        arr (np.ndarray): Input array

    Returns:
        arr (np.ndarray): View of input array with at least two dimensions
    """
    if arr.ndim == 0:
        return arr.reshape((1, 1))
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    return arr


def at_least_3d(arr):
    """View input array as array with at least three dimensions.

    Args:
        arr (np.ndarray): Input array

    Returns:
        arr (np.ndarray): View of input array with at least three dimensions
    """
    if arr.ndim == 0:
        return arr.reshape((1, 1, 1))
    if arr.ndim == 1:
        return arr[:, np.newaxis, np.newaxis]
    if arr.ndim == 2:
        return arr[:, :, np.newaxis]
    return arr
