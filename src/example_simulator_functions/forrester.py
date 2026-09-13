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
"""Forrester function."""

import numpy as np


def forrester(x1: float) -> float:
    """Forrester function.

    Args:
        x1 (float): Input parameter [0.0, 1.0]

    Returns:
        float: Value of the Forrester function
    """
    return float((6.0 * x1 - 2.0) ** 2 * np.sin(12.0 * x1 - 4.0))
