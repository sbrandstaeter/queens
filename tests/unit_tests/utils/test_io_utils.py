"""Test module for the io utils."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from queens.utils.exceptions import FileTypeError
from queens.utils.io_utils import load_input_file, write_to_csv


@pytest.fixture(name="input_dict")
def fixture_input_dict():
    """Input dict for testing."""
    return {"test_key": "test_value"}


@pytest.fixture(name="input_file", params=["json", "yml", "yaml"])
def fixture_input_file(request, input_dict, tmp_path):
    """Input files for testing."""
    file_type = request.param
    input_file_path = tmp_path / f"input_file.{file_type}"
    if file_type == "json":
        dumper = json.dump
    elif file_type in ("yml", "yaml"):
        dumper = yaml.dump
    with open(input_file_path, "w", encoding="utf-8") as stream:
        dumper(input_dict, stream)
    return input_file_path


def test_load_input_file_nonexisting_file():
    """Test if exception is raised if file does not exist."""
    input_path = Path("/fake/file")
    with pytest.raises(FileNotFoundError):
        load_input_file(input_path)


def test_load_input_file_wrong_file_type(tmp_path):
    """Test if an exception is raised for the wrong file type."""
    input_path = tmp_path / "input.file"
    input_path.touch(mode=438)
    with pytest.raises(FileTypeError):
        load_input_file(input_path)


def test_load_input_file(input_file, input_dict):
    """Test *load_input_file*."""
    loaded_dict = load_input_file(input_file)
    assert loaded_dict == input_dict


def test_write_to_csv(tmp_path):
    """Test csv writer."""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # write out data
    output_file_path = Path(tmp_path, "my_csv_file.csv")
    write_to_csv(output_file_path, data)

    # read data from written out file with basic readline routine
    read_in_data_lst = []
    file = output_file_path.read_text(encoding="utf-8")
    read_in_data_lst = [line.strip().split(",") for line in file.splitlines()]

    read_in_data = np.array(read_in_data_lst, dtype=np.float64)

    # read the data in again and compare to original data
    np.testing.assert_array_equal(data, read_in_data)
