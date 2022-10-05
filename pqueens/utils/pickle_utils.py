"""Utils to handle pickle files."""
import pickle
import logging
from pathlib import Path

_logger = logging.getLogger(__name__)

def load_pickle(file_path):
    """Load a pickle file directly from path.

    Args:
        file_path (pathlib.path): Path to pickle-file

    Returns:
        data (dict) in the pickle file.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    try:
        data = pickle.load(open(file_path, "rb"))
        return data
    except Exception as exception:
        raise IOError(f"Could not open the pickle file {file_path}") from exception


def print_pickled_data(file_path):
    """Print a table of the data within a pickle file.

    Only goes one layer deep for dicts. This is similar to `python -m pickle file_path` but makes
     it a single command and fancy prints.

    Args:
        file_path (str): Path to pickle-file
    """
    data = load_pickle(Path(file_path))
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
        _logger.info()


def _create_single_item_string(key, item, item_type, seperator="="):
    """Create a table for a single item.

    Args:
        key (str): Key of the item
        item (obj): Item value for the key
        item_type (str): Type of the item value
        seperator (str, optional): Create seperator line. Defaults to "=".

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
