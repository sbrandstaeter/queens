"""Configuration of folder structure of QUEENS experiments."""
import logging
from pathlib import Path

from pqueens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)

BASE_DATA_DIR = "queens-simulation-data"
EXPERIMENTS_BASE_FOLDER_NAME = "experiments"


def local_base_directory():
    """Base directory holding all queens related data on local machine."""
    base_dir = Path().home() / BASE_DATA_DIR
    create_directory(base_dir)
    return base_dir


def remote_base_directory(remote_connect):
    """Base directory holding all queens related data on remote machine."""
    _, _, remote_home, _ = run_subprocess(
        "echo ~",
        subprocess_type="remote",
        remote_connect=remote_connect,
        additional_error_message=f"Unable to identify home on remote.\n"
        f"Tried to connect to {remote_connect}.",
    )
    base_dir = Path(remote_home.rstrip()) / BASE_DATA_DIR
    create_directory(base_dir, remote_connect=remote_connect)
    return base_dir


def base_directory(remote_connect=None):
    """Base directory holding all queens related data."""
    if remote_connect is None:
        return local_base_directory()

    return remote_base_directory(remote_connect)


def experiments_base_directory(remote_connect=None):
    """Base directory for all experiments on the computing machine."""
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
        subprocess_type = 'simple'
        location = ""
    else:
        subprocess_type = 'remote'
        location = f" on {remote_connect}"

    _logger.debug("Creating folder %s%s.", dir_path, location)
    command_string = f'mkdir -v -p {dir_path}'
    _, _, stdout, _ = run_subprocess(
        command_string=command_string,
        subprocess_type=subprocess_type,
        remote_connect=remote_connect,
    )
    if stdout:
        _logger.debug(stdout)
    else:
        _logger.debug("%s already exists%s.", dir_path, location)


ABS_SINGULARITY_IMAGE_PATH = local_base_directory() / "singularity_image.sif"

LOCAL_TEMPORARY_SUBMISSION_SCRIPT = local_base_directory() / "temporary_submission_script.sh"
