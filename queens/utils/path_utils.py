"""Path utilities for QUEENS."""

from pathlib import Path

PATH_TO_SOURCE = Path(__file__).parents[1]
PATH_TO_QUEENS = Path(__file__).parents[2]


def relative_path_from_source(relative_path):
    """Create relative path from *queens/*.

    As an example to create: *queens/queens/folder/file.A*.

    Call *relative_path_from_source("folder/file.A")* .

    Args:
        relative_path (str): "Path" starting from *queens/queens/*
    Returns:
        PosixPath: Absolute path to the file
    """
    full_path = PATH_TO_SOURCE / relative_path
    return full_path


def relative_path_from_queens(relative_path):
    """Create relative path from *queens/*.

    As an example to create: *queens/queens/folder/file.A* .

    Call *relative_path_from_source("queens/folder/file.A")* .

    Args:
        relative_path (str): "Path" starting from *queens/*

    Returns:
        PosixPath: Absolute path to the file
    """
    full_path = PATH_TO_QUEENS / relative_path
    return full_path


def create_folder_if_not_existent(path):
    """Create folder if not existent.

    Args:
        path (PosixPath): Path to be created

    Returns:
        path_obj (PosixPath): Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def check_if_path_exists(path, error_message=""):
    """Check if a path exists.

    Args:
        path (str): "Path" to be checked
        error_message (str,optional): If an additional message is desired
    Returns:
        path_exists: TODO_doc
    """
    path_exists = Path(path).exists()

    if not path_exists:
        raise FileNotFoundError(error_message + f"\nPath {path} does not exist.")

    return path_exists
