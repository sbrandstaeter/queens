"""Configuration of folder structure of QUEENS experiments."""
import logging
import pathlib

from pqueens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)

BASE_DATA_DIR = "queens-simulation-data"
EXPERIMENTS_BASE_FOLDER_NAME = "experiments"
QUEENS_REPO_BASE_FOLDER_NAME = "repository"


def local_base_directory():
    """Hold all queens related data on local machine."""
    base_dir = pathlib.Path().home() / BASE_DATA_DIR
    create_directory(base_dir)
    return base_dir


def remote_base_directory(remote_connect):
    """Hold all queens related data on remote machine."""
    _, _, remote_home, _ = run_subprocess(
        "echo ~",
        subprocess_type="remote",
        remote_connect=remote_connect,
        additional_error_message=f"Unable to identify home on remote.\n"
        f"Tried to connect to {remote_connect}.",
    )
    base_dir = pathlib.Path(remote_home.rstrip()) / BASE_DATA_DIR
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


def current_job_directory(experiment_dir, job_id):
    """Directory of the latest submitted job.

    Args:
        experiment_dir (pathlib.Path): Experiment directory
        job_id (str): Job ID of the current job

    Returns:
        job_dir (pathlib.Path): Path to the current job directory.
    """
    job_dir = experiment_dir / str(job_id)
    return job_dir


def remote_queens_directory(remote_connect):
    """Hold queens source code on remote machine."""
    queens_directory_on_remote = (
        remote_base_directory(remote_connect=remote_connect) / QUEENS_REPO_BASE_FOLDER_NAME
    )
    create_directory(queens_directory_on_remote, remote_connect=remote_connect)
    return queens_directory_on_remote


ABS_SINGULARITY_IMAGE_PATH = local_base_directory() / "singularity_image.sif"

LOCAL_TEMPORARY_SUBMISSION_SCRIPT = local_base_directory() / "temporary_submission_script.sh"

if __name__ == '__main__':
    # this print statement is intended
    # calling this file as a standalone Python script from the shell will return
    # the absolute path to the singularity image as expected during a QUEENS run.
    # This is needed to make the absolute path to the singularity image available in CI pipelines
    print(f"{ABS_SINGULARITY_IMAGE_PATH}")
