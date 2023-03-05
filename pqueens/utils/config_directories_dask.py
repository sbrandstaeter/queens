"""Configuration of folder structure of QUEENS experiments."""
import logging
import pathlib

_logger = logging.getLogger(__name__)

BASE_DATA_DIR = "queens-simulation-data"
EXPERIMENTS_BASE_FOLDER_NAME = "experiments"


def base_directory():
    """Base directory holding all queens related data on local machine."""
    base_dir = pathlib.Path().home() / BASE_DATA_DIR
    create_directory(base_dir)
    return base_dir


def experiments_base_directory():
    """Base directory for all experiments on the computing machine."""
    base_dir = base_directory()
    experiments_base_dir = base_dir / EXPERIMENTS_BASE_FOLDER_NAME
    create_directory(experiments_base_dir)
    return experiments_base_dir


def experiment_directory(experiment_name):
    """Directory for data of a specific experiment on the computing machine."""
    experiments_base_dir = experiments_base_directory()
    experiment_dir = experiments_base_dir / experiment_name
    create_directory(experiment_dir)
    return experiment_dir


def create_directory(dir_path):
    """Create a directory either local or remote."""
    pathlib.Path.mkdir(dir_path, parents=True, exist_ok=True)
