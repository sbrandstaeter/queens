"""Configuration of folder structure of QUEENS experiments."""
import logging
from pathlib import Path

from pqueens.utils.run_subprocess import run_subprocess_remote

_logger = logging.getLogger(__name__)

BASE_DATA_DIR = "queens-simulation-data"
EXPERIMENTS_BASE_FOLDER_NAME = "experiments"
TESTS_BASE_FOLDER_NAME = "tests"


def local_base_directory():
    """Hold all queens related data on local machine."""
    base_dir = Path().home() / BASE_DATA_DIR
    create_directory(base_dir)
    return base_dir


def remote_home(remote_connect):
    """Get home of remote user."""
    _, _, home, _ = run_subprocess_remote(
        "echo ~",
        remote_connect=remote_connect,
        additional_error_message=f"Unable to identify home on remote.\n"
        f"Tried to connect to {remote_connect}.",
    )
    return Path(home.rstrip())


def remote_base_directory(remote_connect):
    """Hold all queens related data on remote machine."""
    base_dir = remote_home(remote_connect) / BASE_DATA_DIR
    create_directory(base_dir, remote_connect=remote_connect)
    return base_dir


def base_directory(remote_connect=None):
    """Hold all queens related data."""
    if remote_connect is None:
        return local_base_directory()

    return remote_base_directory(remote_connect)


def experiments_base_directory(remote_connect=None):
    """Hold all experiment data on the computing machine."""
    base_dir = base_directory(remote_connect=remote_connect)
    experiments_base_dir = base_dir / EXPERIMENTS_BASE_FOLDER_NAME
    create_directory(experiments_base_dir, remote_connect=remote_connect)
    return experiments_base_dir


def experiment_directory(experiment_name, remote_connect=None):
    """Directory for data of a specific experiment on the computing machine."""
    experiments_base_dir = experiments_base_directory(remote_connect=remote_connect)
    experiment_dir = experiments_base_dir / experiment_name
    create_directory(experiment_dir, remote_connect=remote_connect)
    return experiment_dir


def create_directory(dir_path, remote_connect=None):
    """Create a directory either local or remote."""
    if remote_connect is None:
        location = ""
    else:
        location = f" on {remote_connect}"

    _logger.debug("Creating folder %s%s.", dir_path, location)
    command_string = f'mkdir -v -p {dir_path}'
    _, _, stdout, _ = run_subprocess_remote(command=command_string, remote_connect=remote_connect)
    if stdout:
        _logger.debug(stdout)
    else:
        _logger.debug("%s already exists%s.", dir_path, location)


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
