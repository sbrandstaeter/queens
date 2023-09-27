"""Utils to build queens on remote resource."""
import argparse
import logging
import sys
import time
from pathlib import Path

from pqueens.utils.exceptions import SubprocessError
from pqueens.utils.path_utils import PATH_TO_QUEENS
from pqueens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)

DEFAULT_PACKAGE_MANAGER = "mamba"
FALLBACK_PACKAGE_MANAGER = "conda"
SUPPORTED_PACKAGE_MANAGERS = [DEFAULT_PACKAGE_MANAGER, FALLBACK_PACKAGE_MANAGER]


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


def build_remote_environment(
    remote_address,
    remote_user,
    remote_queens_repository,
    remote_python,
    package_manager=DEFAULT_PACKAGE_MANAGER,
):
    """Build remote QUEENS environment.

    Args:
        remote_address (str): address of remote host
        remote_user (str): remote username
        remote_queens_repository (str): Path to queens repository on remote host
        remote_python (str): Path to Python environment on remote host
        package_manager(str, optional): Package manager used for the creation of the environment:
                                        "mamba" or "conda"
    """
    if package_manager not in SUPPORTED_PACKAGE_MANAGERS:
        raise ValueError(
            f"The package manager '{package_manager}' is not supported.\n"
            f"Supported package managers are: {SUPPORTED_PACKAGE_MANAGERS}"
        )
    remote_connect = f'{remote_user}@{remote_address}'

    # check if requested package_manager is installed on remote machine:
    def package_manager_exists_remote(package_manager_name):
        """Check if requested package manager exists on remote.

        Args:
            package_manager_name (string): name of package manager
        """
        try:
            run_subprocess(
                f'which {package_manager_name}',
                remote_connect=remote_connect,
            )
        except SubprocessError as error:
            message = (
                f"Could not find requested package manager '{package_manager_name}' "
                f"on '{remote_connect}'."
            )
            if package_manager_name == DEFAULT_PACKAGE_MANAGER:
                _logger.warning(message)
                _logger.warning(
                    "Trying to fall back to the '%s' package manager.", FALLBACK_PACKAGE_MANAGER
                )
                package_manager_exists_remote(package_manager_name=FALLBACK_PACKAGE_MANAGER)
            else:
                raise SubprocessError(message) from error
            return False
        return True

    if not package_manager_exists_remote(package_manager_name=package_manager):
        package_manager = FALLBACK_PACKAGE_MANAGER

    _logger.info("Build remote QUEENS environment...")
    environment_name = Path(remote_python).parents[1].name
    command_string = (
        f'cd {remote_queens_repository}; '
        f'{package_manager} env create -f environment.yml --name {environment_name} --force; '
        f'{package_manager} activate {environment_name};'
        f'pip install -e .'
    )
    start_time = time.time()
    _, _, stdout, _ = run_subprocess(
        command_string,
        remote_connect=remote_connect,
        raise_error_on_subprocess_failure=False,
    )
    _logger.debug(stdout)
    _logger.info("Build of remote queens environment was successful.")
    _logger.info("It took: %s s.\n", time.time() - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build queens environment on remote machine.")
    parser.add_argument(
        '--remote-address', type=str, required=True, help='hostname or ip address of remote host'
    )
    parser.add_argument('--remote-user', type=str, required=True, help='remote username')
    parser.add_argument(
        '--remote-queens-repository',
        type=str,
        required=True,
        help='path to queens repository on remote host',
    )
    parser.add_argument(
        '--remote-python', type=str, required=True, help='path to python environment on remote host'
    )
    parser.add_argument(
        '--package-manager',
        type=str,
        default=DEFAULT_PACKAGE_MANAGER,
        choices=SUPPORTED_PACKAGE_MANAGERS,
        help='package manager used for the creation of the remote environment',
    )

    args = parser.parse_args(sys.argv[1:])

    sync_remote_repository(args.remote_address, args.remote_user, args.remote_queens_repository)
    build_remote_environment(
        args.remote_address,
        args.remote_user,
        args.remote_queens_repository,
        args.remote_python,
        args.package_manager,
    )
    sys.exit()
