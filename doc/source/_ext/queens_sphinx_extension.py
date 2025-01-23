#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2025, QUEENS contributors.
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
"""This is a custom extension to create files for sphinx using python."""

import os
import sys

from sphinx.application import Sphinx

sys.path.insert(0, os.path.abspath("."))

import create_documentation_files  # pylint: disable=wrong-import-position


def run_custom_code(app: Sphinx):  # pylint: disable=unused-argument
    """Run the custom code."""
    create_documentation_files.main()


def setup(app: Sphinx):  # pylint: disable=unused-argument
    """Setup up sphinx app."""
    app.connect("builder-inited", run_custom_code)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
    }
