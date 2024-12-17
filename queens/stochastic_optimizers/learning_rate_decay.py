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
"""Learning rate decay for stochastic optimization."""

import abc
import logging

_logger = logging.getLogger(__name__)


class LearningRateDecay(metaclass=abc.ABCMeta):
    """Base class for learning rate decay."""

    @abc.abstractmethod
    def __call__(self, learning_rate, params, gradient):
        """Adapt learning rate.

        Args:
            learning_rate (float): Current learning rate
            params (np.array): Current parameters
            gradient (np.array): Current gradient

        Returns:
            learning_rate (float): Adapted learning rate
        """


class LogLinearLearningRateDecay(LearningRateDecay):
    """Log linear learning rate decay.

    Attributes:
        slope (float): Logarithmic slope
        iteration (int): Current iteration
    """

    def __init__(self, slope):
        """Initialize LogLinearLearningRateDecay.

        Args:
            slope (float): Logarithmic slope
        """
        self.slope = slope
        self.iteration = 0

    def __call__(self, learning_rate, params, gradient):
        """Adapt learning rate.

        Args:
            learning_rate (float): Current learning rate
            params (np.array): Current parameters
            gradient (np.array): Current gradient

        Returns:
            learning_rate (float): Adapted learning rate
        """
        self.iteration += 1
        learning_rate /= self.iteration**self.slope
        return learning_rate
