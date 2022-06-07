"""Path utilities for QUEENS."""
from pathlib import Path

PATH_TO_PQUEENS = Path(__file__).parents[1]
PATH_TO_QUEENS = Path(__file__).parents[2]


def relative_path_from_pqueens(relative_path):
    """Create relative path from `pqueens/`.

    As an example to create:
        queens/pqueens/folder/file.A

    call relative_path_from_pqueens("folder/file.A")

    Args:
        relative_path (str): Path starting from queens/pqueens/

    Returns:
        (str): Absolute path to the file
    """
    return PATH_TO_PQUEENS.joinpath(relative_path)


def relative_path_from_queens(relative_path):
    """Create relative path from `queens/`.

    As an example to create:
        queens/pqueens/folder/file.A

    call relative_path_from_pqueens("pqueens/folder/file.A")

    Args:
        relative_path (str): Path starting from queens/

    Returns:
        (str): Absolute path to the file
    """
    return PATH_TO_QUEENS.joinpath(relative_path)


def create_folder_if_not_existent(path):
    """Create folder if not existent.

    Args:
        path (PosixPath): Path to be created

    Returns:
        path_obj (PosixPath) path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def check_if_path_exists(path, error_message=""):
    """Check if a path exists.

    Args:
        path (str): Path to be checked
        error_message (str,optional): If an additional message is desired
    """
    path_exists = Path(path).exists()

    if not path_exists:
        raise FileNotFoundError(error_message + f"\nPath {path} does not exist.")

    return path_exists
