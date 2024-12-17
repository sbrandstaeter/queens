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
"""Test-module for tempering functionality of smc_utils module.

@author: Sebastian Brandstaeter
"""

import math

import numpy as np
import pytest

from queens.utils import smc_utils


@pytest.fixture(
    name="temper_keyword_and_temper_type",
    scope="module",
    params=[
        ("bayes", type(smc_utils.temper_logpdf_bayes)),
        ("generic", type(smc_utils.temper_logpdf_generic)),
    ],
)
def fixture_temper_keyword_and_temper_type(request):
    """Return a set of valid keyword and corresponding temper type."""
    return request.param


@pytest.fixture(name="temper_parameter", scope="module", params=[0.0, 1e-4, 0.9, 1.0])
def fixture_temper_parameter(request):
    """Return a valid temper parameter."""
    return request.param


@pytest.fixture(name="logpdf0", scope="module", params=[-np.inf, -1e8, 0.0, 1e10])
def fixture_logpdf0(request):
    """Return a valid logpdf."""
    return request.param


@pytest.fixture(name="logpdf1", scope="module", params=[-np.inf, -1e8, 0.0, 1e10])
def fixture_logpdf1(request):
    """Return a valid logpdf."""
    return request.param


def test_temper_factory(temper_keyword_and_temper_type):
    """Test the function factory for valid tempering functions."""
    temper, temper_type_sol = temper_keyword_and_temper_type
    temper_type = smc_utils.temper_factory(temper)

    assert isinstance(temper_type, temper_type_sol)


def test_temper_factory_invalid():
    """Test the function factory for invalid keyword."""
    temper = "invalid"
    with pytest.raises(ValueError, match=r"Unknown type.*"):
        smc_utils.temper_factory(temper)


def test_temper_logpdf_bayes(logpdf0, logpdf1, temper_parameter):
    """Test the bayesian tempering function."""
    sum1 = temper_parameter * logpdf1
    if math.isclose(temper_parameter, 0.0, abs_tol=1e-8):
        sum1 = 0.0

    tempered_logpdf_sol = sum1 + logpdf0
    tempered_logpdf = smc_utils.temper_logpdf_bayes(logpdf0, logpdf1, temper_parameter)
    assert np.isclose(tempered_logpdf, tempered_logpdf_sol)


def test_temper_logpdf_bayes_posinf_invalid(logpdf0, temper_parameter):
    """Test the bayesian tempering function is invalid for infinity."""
    with pytest.raises(ValueError):
        smc_utils.temper_logpdf_bayes(np.inf, logpdf0, temper_parameter)
    with pytest.raises(ValueError):
        smc_utils.temper_logpdf_bayes(logpdf0, np.inf, temper_parameter)


def test_temper_logpdf_generic(logpdf0, logpdf1, temper_parameter):
    """Test the generic tempering function."""
    if math.isclose(temper_parameter, 0.0, abs_tol=1e-8):
        tempered_logpdf_sol = logpdf0
    elif math.isclose(temper_parameter, 1.0):
        tempered_logpdf_sol = logpdf1
    else:
        tempered_logpdf_sol = temper_parameter * logpdf1 + (1.0 - temper_parameter) * logpdf0

    tempered_logpdf = smc_utils.temper_logpdf_generic(logpdf0, logpdf1, temper_parameter)
    assert np.isclose(tempered_logpdf, tempered_logpdf_sol)


def test_temper_logpdf_generic_posinf_invalid(logpdf0, temper_parameter):
    """Test the generic tempering function is invalid for infinity."""
    with pytest.raises(ValueError):
        smc_utils.temper_logpdf_generic(np.inf, logpdf0, temper_parameter)
    with pytest.raises(ValueError):
        smc_utils.temper_logpdf_generic(logpdf0, np.inf, temper_parameter)
