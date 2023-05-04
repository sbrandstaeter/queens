"""Utils to build queens on remote resource."""
import argparse
import logging
import sys
import time
from pathlib import Path

from pqueens.utils.exceptions import CLIError
from pqueens.utils.path_utils import PATH_TO_QUEENS
from pqueens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)


def sync_remote_repository(remote_address, remote_user, remote_queens_repository):
    """Synchronize local and remote QUEENS source files.

    Args:
        remote_address (str): address of remote host
        remote_user (str): remote username
        remote_queens_repository (str): Path to queens repository on remote host
    """
    _logger.info("Syncing remote QUEENS repository with local one...")
    run_subprocess(f"ssh {remote_user}@{remote_address} mkdir -p {remote_queens_repository}")
    command_list = [
        "rsync --out-format='%n' --archive --checksum --verbose --verbose",
        "--filter=':- .gitignore' --exclude '.git'",
        f"{PATH_TO_QUEENS}/",
        f"{remote_user}@{remote_address}:{remote_queens_repository}",
    ]
    command_string = ' '.join(command_list)
    start_time = time.time()
    _, _, stdout, _ = run_subprocess(
        command_string,
        additional_error_message="Error during sync of local and remote QUEENS repositories! ",
    )
    _logger.debug(stdout)
    _logger.info("Sync of remote repository was successful.")
    _logger.info("It took: %s s.\n", time.time() - start_time)


def build_remote_environment(remote_address, remote_user, remote_queens_repository, remote_python):
    """Build remote QUEENS environment.

    Args:
        remote_address (str): address of remote host
        remote_user (str): remote username
        remote_queens_repository (str): Path to queens repository on remote host
        remote_python (str): Path to Python environment on remote host
    """
    _logger.info("Build remote QUEENS environment...")
    environment_name = Path(remote_python).parents[1].name
    command_string = (
        f'ssh {remote_user}@{remote_address} "'
        f'cd {remote_queens_repository}; '
        f'conda env create -f environment.yml --name {environment_name} --force; '
        f'conda activate {environment_name};'
        f'pip install -e ."'
    )
    start_time = time.time()
    _, _, stdout, _ = run_subprocess(command_string)
    _logger.debug(stdout)
    _logger.info("Build of remote queens environment was successful.")
    _logger.info("It took: %s s.\n", time.time() - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build queens environment on remote machine.")
    parser.add_argument('--remote_address', type=str, default=None, help='address of remote host')
    parser.add_argument('--remote_user', type=str, default=None, help='remote username')
    parser.add_argument(
        '--remote_queens_repository',
        type=str,
        default=None,
        help='path to queens repository on remote host',
    )
    parser.add_argument(
        '--remote_python', type=str, default=None, help='path to python environment on remote host'
    )

    args = parser.parse_args(sys.argv[1:])

    for arg in vars(args).values():
        if arg is None:
            raise CLIError(f"Missing option --{arg}.")

    sync_remote_repository(args.remote_address, args.remote_user, args.remote_queens_repository)
    build_remote_environment(
        args.remote_address, args.remote_user, args.remote_queens_repository, args.remote_python
    )
    sys.exit()
