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
"""Extract QUEENS example from the readme."""


from queens.utils.path_utils import relative_path_from_queens

EXAMPLE_MARKER = "<!---example marker, do not remove this comment-->"


def get_queens_example_from_readme(output_dir):
    """Extract the example from the readme.

    Args:
        output_dir (str): Output directory for the QUEENS run.
    """
    readme_path = relative_path_from_queens("README.md")

    # Split the example in the readme using the marker
    text = readme_path.read_text().split(EXAMPLE_MARKER)

    # Only one example should appear
    if len(text) != 3:
        raise ValueError("Could not extract the example from the readme!")

    # Extract the source
    example_source = (
        text[1]
        .replace("```python", "")
        .replace("```", "")
        .replace('output_dir="."', f'output_dir="{output_dir}"')
    )

    return example_source
