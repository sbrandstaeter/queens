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
"""Goldstein-Price function."""

import numpy as np


def goldstein_price(x1: float, x2: float) -> float:
    """Goldstein-Price function.

    Args:
        x1 (float): Input parameter [-2.0, 2.0]
        x2 (float): Input parameter [-2.0, 2.0]

    Returns:
        float: Value of the Goldstein-Price function
    """
    factor_1 = 1.0 + np.square(x1 + x2 + 1.0) * (
        19.0 - 14.0 * x1 + 3.0 * np.square(x1) - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * np.square(x2)
    )

    factor_2 = 30.0 + np.square(2.0 * x1 - 3.0 * x2) * (
        18.0 - 32.0 * x1 + 12.0 * np.square(x1) + 48.0 * x2 - 36.0 * x1 * x2 + 27.0 * np.square(x2)
    )

    return float(factor_1 * factor_2)
