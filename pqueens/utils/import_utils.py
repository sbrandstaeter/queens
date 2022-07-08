"""Import utils."""
import importlib.util
import sys
from pathlib import Path

from pqueens.utils.path_utils import check_if_path_exists


def load_function_or_class_by_name_from_path(path_to_module, function_or_class_name):
    """Load function from python file by path.

    Args:
        path_to_module (str): Path to file
        function_or_class_name (str): Name of the function

    Returns:
        (function or class): function or class from the module
    """
    # Set the module name
    module_path_obj = Path(path_to_module)
    module_name = module_path_obj.stem
    module_ending = module_path_obj.suffix

    # Check if file exsits
    if not check_if_path_exists(module_path_obj):
        raise FileNotFoundError(f"Could not find python file {path_to_module}.")

    # Check if ending is correct
    if module_ending != ".py":
        raise ImportError(f"Python file {path_to_module} does not have a .py ending")

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, path_to_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module

    try:
        # Check if function can be loaded
        function = getattr(module, function_or_class_name)
    except AttributeError as error:
        raise AttributeError(
            f"External python module {path_to_module} does not have a function or class"
            f"{function_or_class_name}"
        ) from error

    return function
