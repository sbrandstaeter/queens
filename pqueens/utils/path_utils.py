"""Path utilities for QUEENS."""
import os

PATH_TO_PQUEENS = os.path.join(os.path.dirname(__file__), "../")
PATH_TO_QUEENS = os.path.join(os.path.dirname(__file__), "../../")


def relative_path_from_pqueens(relative_path):
    """Create relative path from `pqueens/`.

    As an example to create:
        queens/pqueens/folder/file.A

    call relative_path_from_pqueens("folder/file.A")

    Args:
        relative_path (str): Path starting from queens/pqueens/

    Returns:
        [str]: Absolute path to the file
    """
    return os.path.join(PATH_TO_PQUEENS, relative_path)


def relative_path_from_queens(relative_path):
    """Create relative path from `queens/`.

    As an example to create:
        queens/pqueens/folder/file.A

    call relative_path_from_pqueens("pqueens/folder/file.A")

    Args:
        relative_path (str): Path starting from queens/

    Returns:
        [str]: Absolute path to the file
    """
    return os.path.join(PATH_TO_QUEENS, relative_path)


def create_folder_if_not_existent(path):
    """Create folder if not existent.

    Args:
        path (str): Path to be created
    """
    os.makedirs(path, exist_ok=True)
