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
"""Utils to handle pickle files."""

import logging
import pickle

_logger = logging.getLogger(__name__)


def load_pickle(file_path):
    """Load a pickle file directly from path.

    Args:
        file_path (Path): Path to pickle-file

    Returns:
        Data (dict) in the pickle file
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    try:
        data = pickle.load(file_path.open("rb"))
        return data
    except Exception as exception:
        raise IOError(f"Could not open the pickle file {file_path}") from exception


def print_pickled_data(file_path):
    """Print a table of the data within a pickle file.

    Only goes one layer deep for dicts. This is similar to *python -m pickle file_path* but makes
    it a single command and fancy prints.

    Args:
        file_path (Path): Path to pickle-file
    """
    data = load_pickle(file_path)
    _logger.info("\n\npickle file: %s", file_path)
    for key, item in data.items():
        item_type = type(item)
        if isinstance(item, dict):
            string = ""
            for subkey, subitem in item.items():
                string += (
                    _create_single_item_string(subkey, subitem, type(subitem), seperator="-") + "\n"
                )
            item = string.replace("\n", "\n    ")
        _logger.info(_create_single_item_string(key, item, item_type))
        _logger.info(" ")


def _create_single_item_string(key, item, item_type, seperator="="):
    """Create a table for a single item.

    Args:
        key (str): Key of the item
        item (obj): Item value for the key
        item_type (str): Type of the item value
        seperator (str, optional): Create seperator line (default is "=")

    Returns:
        string: table for this item.
    """
    string = (
        seperator * 60
        + f"\nKey:  {key}\n"
        + f"Type: {item_type}\n"
        + f"Value:\n{item}\n"
        + seperator * 60
    )
    return string
