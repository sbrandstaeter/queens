"""Configuration of folder structure of QUEENS experiments."""
import logging
import pathlib

from pqueens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)

BASE_DATA_DIR = "queens-experiments"


def local_base_dir():
    """Base directory holding all queens related data on local machine."""
    base_dir = pathlib.Path().home() / BASE_DATA_DIR
    create_directory(base_dir)
    return base_dir


def remote_base_dir(remote_connect):
    """Base directory holding all queens related data on remote machine."""
    _, _, remote_home, _ = run_subprocess(
        "echo $HOME",
        subprocess_type="remote",
        remote_connect=remote_connect,
        additional_error_message=f"Unable to identify home on remote.\n"
        f"Tried to connect to {remote_connect}.",
    )
    base_dir = pathlib.Path(remote_home.rstrip()) / BASE_DATA_DIR
    create_directory(base_dir, remote_connect=remote_connect)
    return base_dir


def experiment_directory(experiment_name, remote_connect=None):
    """Directory for data of an experiment."""
    if remote_connect is None:
        base_dir = local_base_dir()
    else:
        base_dir = remote_base_dir(remote_connect)
    experiment_dir = base_dir / experiment_name
    create_directory(experiment_dir, remote_connect=remote_connect)
    return experiment_dir


def create_directory(dir_path, remote_connect=None):
    """Create a directory either local or remote."""
    if remote_connect is not None:
        subprocess_type = 'remote'
        location = ""
    else:
        subprocess_type = 'simple'
        location = f" on {remote_connect}"

    _logger.debug(f"Creating folder {dir_path}{location}.")
    command_string = f'mkdir -v -p {dir_path}'
    _, _, stdout, _ = run_subprocess(
        command_string=command_string,
        subprocess_type=subprocess_type,
        remote_connect=remote_connect,
    )
    if stdout:
        _logger.debug(stdout)
    else:
        _logger.debug(f"{dir_path} already exists{location}.")
