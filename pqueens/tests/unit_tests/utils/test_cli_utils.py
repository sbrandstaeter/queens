"""Tests for cli utils."""
from pathlib import Path

import pytest

from pqueens.utils.cli_utils import get_cli_options, DEFAULT_DASK_SCHEDULER_PORT
from pqueens.utils.exceptions import CLIError


@pytest.fixture(name="debug_flag", params=[True, False])
def fixture_debug_flag(request):
    """Debug flag."""
    return request.param


def test_get_cli_options_no_input():
    """Test if no input is provided."""
    args = ["--output_dir", "output_dir"]
    with pytest.raises(CLIError, match="No input file was provided with option --input"):
        get_cli_options(args)


def test_get_cli_options_no_output():
    """Test if no output is provided."""
    args = ["--input", "input_file"]
    with pytest.raises(
        CLIError, match="No output directory was provided with option --output_dir."
    ):
        get_cli_options(args)


def test_get_cli_options_default():
    """Test if default debug option is set correctly."""
    args = ["--input", "input_file", "--output_dir", "output_dir"]
    _, _, debug, dask_scheduler_port = get_cli_options(args)
    assert not debug
    assert dask_scheduler_port == DEFAULT_DASK_SCHEDULER_PORT


def test_get_cli_options_debug(debug_flag):
    """Test if options are read in correctly."""
    goal_dask_scheduler_port = 12345
    args = [
        "--input",
        "input_file",
        "--output_dir",
        "output_dir",
        "--debug",
        str(debug_flag),
        "--dask-scheduler-port",
        str(goal_dask_scheduler_port),
    ]
    input_file, output_dir, debug, dask_scheduler_port = get_cli_options(args)
    assert input_file == Path("input_file")
    assert output_dir == Path("output_dir")
    assert debug == debug_flag
    assert dask_scheduler_port == goal_dask_scheduler_port
