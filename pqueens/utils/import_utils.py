"""Import utils."""
import importlib.util
import sys
from pathlib import Path

from pqueens.utils.path_utils import check_if_path_exists


def load_main_by_path(path_to_module):
    """Load a main function from python file by path.

    Args:
        path_to_module (str): Path to file

    Returns:
        (function): Main function of the module
    """
    if not check_if_path_exists(path_to_module):
        raise FileNotFoundError(f"Could not find python file {path_to_module}.")

    # Set the module name
    module_name = Path(path_to_module).stem

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, path_to_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module

    # Return its main function
    return module.main
