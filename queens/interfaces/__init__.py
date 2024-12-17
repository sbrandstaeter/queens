# -*- coding: utf-8 -*-
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
"""Interfaces.

This package contains a set of so-called interfaces. The purpose of an
interface is essentially the mapping of inputs to outputs.

The mapping is made by passing the inputs further down to a
*regression_approximation* or a *mf_regression_approximation*, both of
which essentially then evaluate a regression model themselves.
"""

from queens.interfaces.bmfia_interface import BmfiaInterface

VALID_TYPES = {
    "bmfia_interface": BmfiaInterface,
}
