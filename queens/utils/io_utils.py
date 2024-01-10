"""Utils for input/output handling."""
import csv
from pathlib import Path

import yaml

from queens.utils.exceptions import FileTypeError
from queens.utils.pickle_utils import load_pickle

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
    with open(input_file_path, "r", encoding='utf-8') as stream:
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
        data (np.array): Data in form of numpy arrays
        delimiter (optional, str): Delimiter to separate individual data.
                                   Defaults to comma delimiter.
    """
    # Write data to new file
    with open(output_file_path, 'w', newline='', encoding='utf-8') as out_file:
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
    file = Path(file_path).read_text(encoding='utf-8')
    return file
