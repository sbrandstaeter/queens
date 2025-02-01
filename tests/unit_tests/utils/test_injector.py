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
"""Test injector util."""

import pytest
from jinja2.exceptions import UndefinedError

from queens.utils.injector import render_template


@pytest.mark.parametrize(
    "injection_parameters",
    [
        {"parameter_1": 1},
        {"parameter_1": 1, "wrong_parameter": 3},
    ],
)
def test_failure_injection(injection_parameters):
    """Test if injection raises an error."""
    template = "{{ parameter_1 }} {{ parameter_2 }}"

    with pytest.raises(UndefinedError):
        render_template(injection_parameters, template)


@pytest.mark.parametrize(
    "injection_parameters,expected_result",
    [
        ({"parameter_1": 1}, "1 "),
        ({"parameter_1": 1, "wrong_parameter": 3}, "1 "),
        ({"parameter_1": 1, "parameter_2": 2, "wrong_parameter": 3}, "1 2"),
    ],
)
def test_injection_strict_is_false(injection_parameters, expected_result):
    """Test injection with strict is False."""
    template = "{{ parameter_1 }} {{ parameter_2 }}"

    obtained_output = render_template(injection_parameters, template, strict=False)
    assert obtained_output == expected_result
