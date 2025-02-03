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
"""Unit tests for LearningRateDecay classes."""

import numpy as np
import pytest

from queens.stochastic_optimizers.learning_rate_decay import (
    DynamicLearningRateDecay,
    LogLinearLearningRateDecay,
    StepwiseLearningRateDecay,
)


def test_init_log_linear_learning_rate_decay():
    """Test the init method of LogLinearLearningRateDecay class."""
    slope = 0.7
    learning_rate_decay = LogLinearLearningRateDecay(slope=slope)
    assert learning_rate_decay.slope == slope
    assert learning_rate_decay.iteration == 0


def test_log_linear_learning_rate_decay():
    """Test the call method of LogLinearLearningRateDecay class."""
    learning_rate = 1.0
    learning_rate_decay = LogLinearLearningRateDecay(slope=0.5)
    num_iter = 101
    for _ in range(num_iter):
        learning_rate = learning_rate_decay(learning_rate, None, None)

    assert learning_rate_decay.iteration == num_iter
    np.testing.assert_array_almost_equal(learning_rate, 0.1)


def test_init_stepwise_learning_rate_decay():
    """Test the init method of StepwiseLearningRateDecay class."""
    decay_factor = 0.1
    decay_interval = 100
    learning_rate_decay = StepwiseLearningRateDecay(
        decay_factor=decay_factor, decay_interval=decay_interval
    )
    assert learning_rate_decay.decay_factor == decay_factor
    assert learning_rate_decay.decay_interval == decay_interval
    assert learning_rate_decay.iteration == 0


def test_stepwise_learning_rate_decay():
    """Test the call method of StepwiseLearningRateDecay class."""
    learning_rate = 1.0
    learning_rate_decay = StepwiseLearningRateDecay(decay_factor=0.1, decay_interval=10)
    for _ in range(25):
        learning_rate = learning_rate_decay(learning_rate, None, None)

    assert learning_rate_decay.iteration == 3
    np.testing.assert_array_almost_equal(learning_rate, 0.01)


def test_init_dynamic_learning_rate_decay():
    """Test the init method of DynamicLearningRateDecay class."""
    alpha = 0.2
    rho_min = 2.0
    learning_rate_decay = DynamicLearningRateDecay(alpha=alpha, rho_min=rho_min)
    assert learning_rate_decay.alpha == alpha
    assert learning_rate_decay.rho_min == rho_min
    assert learning_rate_decay.k_min == 2
    assert learning_rate_decay.k == -1
    assert learning_rate_decay.a == 0
    assert learning_rate_decay.b == 0
    assert learning_rate_decay.c == 0


@pytest.mark.parametrize("alpha", [-1, 0, 1, 2])
def test_init_dynamic_learning_rate_decay_invalid_alpha(alpha):
    """Test the init method of DynamicLearningRateDecay class.

    Test invalid values for alpha.
    """
    with pytest.raises(ValueError):
        DynamicLearningRateDecay(alpha=alpha)


@pytest.mark.parametrize("rho_min", [-1, 0])
def test_init_dynamic_learning_rate_decay_invalid_rho_min(rho_min):
    """Test the init method of DynamicLearningRateDecay class.

    Test invalid values for rho_min.
    """
    with pytest.raises(ValueError):
        DynamicLearningRateDecay(rho_min=rho_min)


def test_dynamic_learning_rate_decay():
    """Test the call method of DynamicLearningRateDecay class."""
    np.random.seed(1)
    learning_rate = 1.0
    learning_rate_decay = DynamicLearningRateDecay()
    params = np.array([1.0, 2.0, 3.0])
    num_iter = 101
    for _ in range(num_iter):
        learning_rate = learning_rate_decay(learning_rate, params, None)
        params += np.random.randn(3) * 0.1

    np.testing.assert_array_almost_equal(learning_rate, 0.01)
