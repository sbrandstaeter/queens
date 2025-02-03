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
"""Test custom errors."""

import pytest

from queens.utils.exceptions import SubprocessError
from queens.utils.run_subprocess import run_subprocess


def test_subprocess_correct_message_construction():
    """Check if SubprocessError message is constructed correctly."""
    command = "dummy command"
    command_output = "dummy command output"
    error_message = "dummy error message"
    additional_message = "additional error message"
    sp_error = SubprocessError.construct_error_from_command(command, command_output, error_message)

    expected_message = (
        "\n\nQUEENS' subprocess wrapper caught the following error:\ndummy"
        " error message\n\n\nwith commandline output:\ndummy command output\n\n\nwhile executing"
        " the command:\ndummy command"
    )
    assert expected_message == str(sp_error)

    expected_message += "\n\nadditional error message"
    sp_error = SubprocessError.construct_error_from_command(
        command, command_output, error_message, additional_message
    )
    assert expected_message == str(sp_error)


def test_subprocess_raises_error():
    """Check if non existing command raises an SubprocessError."""
    with pytest.raises(SubprocessError):
        run_subprocess("NonExistingCommand")
