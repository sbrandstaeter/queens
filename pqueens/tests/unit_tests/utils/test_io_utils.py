"""Test module for the io utils."""

import json
from pathlib import Path

import pytest
import yaml

from pqueens.utils.exceptions import FileTypeError
from pqueens.utils.io_utils import load_input_file

pytestmark = pytest.mark.unit_tests


@pytest.fixture(name="input_dict")
def fixture_input_dict():
    """Input dict for testing."""
    return {"test_key": "test_value"}


@pytest.fixture(name="input_file", params=["json", "yml", "yaml"])
def fixture_input_file(request, input_dict, test_path):
    """Input files for testing."""
    file_type = request.param
    input_file_path = test_path.joinpath("input_file." + file_type)
    if file_type == "json":
        dumper = json.dump
    elif file_type in ("yml", "yaml"):
        dumper = yaml.dump
    with open(input_file_path, "w") as stream:
        dumper(input_dict, stream)
    return input_file_path


def test_load_input_file_nonexisting_file():
    """Test if exception is raised if file does not exist."""
    input_path = Path("/fake/file")
    with pytest.raises(FileNotFoundError):
        load_input_file(input_path)


def test_load_input_file_wrong_file_type(test_path):
    """Test if exception is raised for wrong file type."""
    input_path = test_path.joinpath("input.file")
    open(input_path, "a+")
    with pytest.raises(FileTypeError):
        load_input_file(input_path)


def test_load_input_file(input_file, input_dict):
    """Test load_input_file."""
    loaded_dict = load_input_file(input_file)
    assert loaded_dict == input_dict
