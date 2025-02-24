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
"""Utils for input/output handling."""

import csv
from pathlib import Path

import yaml

from queens.utils.exceptions import FileTypeError
from queens.utils.pickle import load_pickle

try:
    import simplejson as json
except ImportError:
    import json


def load_input_file(input_file_path):
    """Load inputs from file by path.

    Args:
        input_file_path (Path): Path to the input file

    Returns:
        dict: Options in the input file.
    """
    if not input_file_path.is_file():
        raise FileNotFoundError(f"Input file {input_file_path} does not exist.")

    file_type = input_file_path.suffix.lower()
    if file_type == ".json":
        loader = json.load
    elif file_type in [".yml", ".yaml"]:
        loader = yaml.safe_load
    else:
        raise FileTypeError(
            f"Only json or yaml/yml files allowed, not of type '{file_type}' ({input_file_path})"
        )
    with open(input_file_path, "r", encoding="utf-8") as stream:
        try:
            options = loader(stream)
        except Exception as exception:
            raise type(exception)(
                f"Could not load file {input_file_path}. Verify the syntax."
            ) from exception
    return options


def load_result(path_to_result_file):
    """Load QUEENS results.

    Args:
        path_to_result_file (Path): Path to results
    Returns:
        dict: Results
    """
    path_to_result_file = Path(path_to_result_file)
    results = load_pickle(path_to_result_file)
    return results


def write_to_csv(output_file_path, data, delimiter=","):
    """Write a simple csv file.

    Write data out to a csv-file. Nothing fancy, at the moment,
    only now header line or index column is supported just pure data.

    Args:
        output_file_path (Path obj): Path to the file the data should be written to
        data (np.array): Data that should be written to the csv file.
        delimiter (optional, str): Delimiter to separate individual data.
                                   Defaults to comma delimiter.
    """
    # Write data to new file
    with open(output_file_path, "w", newline="", encoding="utf-8") as out_file:
        writer = csv.writer(out_file, delimiter=delimiter)
        # write only new data
        for row in data:
            writer.writerow(row)


def read_file(file_path):
    """Function to read in a file.

    Args:
        file_path (str, Path): Path to file
    Returns:
        file (str): read in file as string
    """
    file = Path(file_path).read_text(encoding="utf-8")
    return file


def to_dict_with_standard_types(obj):
    """Convert dictionaries to dictionaries with python standard types only.

    Args:
        obj (dict): Dictionary to convert

    Returns:
        dict: Dictionary with standard types
    """
    match obj:
        case Path():
            return str(obj)
        case tuple():
            return [to_dict_with_standard_types(value) for value in obj]
        case list():
            return [to_dict_with_standard_types(value) for value in obj]
        case dict():
            for key, value in obj.items():
                obj[key] = to_dict_with_standard_types(value)
            return obj
        case _ if hasattr(obj, "tolist"):
            return obj.tolist()
        case _:
            return obj
