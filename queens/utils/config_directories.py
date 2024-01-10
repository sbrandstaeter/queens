"""Configuration of folder structure of QUEENS experiments."""
import logging
from pathlib import Path

from queens.utils.path_utils import create_folder_if_not_existent

_logger = logging.getLogger(__name__)

BASE_DATA_DIR = "queens-simulation-data"
EXPERIMENTS_BASE_FOLDER_NAME = "experiments"
TESTS_BASE_FOLDER_NAME = "tests"


def base_directory():
    """Hold all queens related data."""
    base_dir = Path().home() / BASE_DATA_DIR
    create_directory(base_dir)
    return base_dir


def experiments_base_directory():
    """Hold all experiment data on the computing machine."""
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
    _logger.debug("Creating folder %s.", dir_path)
    create_folder_if_not_existent(dir_path)


def current_job_directory(experiment_dir, job_id):
    """Directory of the latest submitted job.

    Args:
        experiment_dir (Path): Experiment directory
        job_id (str): Job ID of the current job

    Returns:
        job_dir (Path): Path to the current job directory.
    """
    job_dir = experiment_dir / str(job_id)
    return job_dir
