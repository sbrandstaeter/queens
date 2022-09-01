"""Utils for input/output handling."""
from pathlib import Path

import yaml

from pqueens.utils.exceptions import FileTypeError
from pqueens.utils.pickle_utils import load_pickle

try:
    import simplejson as json
except ImportError:
    import json


def load_input_file(input_file_path):
    """Load inputs from file by path.

    Args:
        input_file_path (pathlib.Path): Path to the input file

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
    with open(input_file_path, "r") as stream:
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
        path_to_result_file (Pathlib.Path): Path to results
    Returns:
        dict: results
    """
    path_to_result_file = Path(path_to_result_file)
    results = load_pickle(path_to_result_file)
    return results
