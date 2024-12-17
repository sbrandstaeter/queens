#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024, QUEENS contributors.
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
"""Utility methods used by the unit tests for distributions."""

import numpy as np


def covariance_discrete(probabilities, sample_space, mean):
    """Compute the covariance of a discrete distribution.

    Args:
        probabilities (np.ndarray): Probability of each event in the sample space
        sample_space (lst): Sample space of the discrete distribution holding the events
        mean (np.ndarray): Mean of the distribution

    Returns:
        np.ndarray: covariance
    """
    return np.sum(
        [
            probability * np.outer(value, value)
            for probability, value in zip(probabilities, sample_space)
        ],
        axis=0,
    ) - np.outer(mean, mean)
