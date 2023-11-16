"""Utils to build queens on remote resource."""
import argparse
import json
import logging
import sys

from queens.utils.remote_operations import RemoteConnection

DEFAULT_PACKAGE_MANAGER = "mamba"
FALLBACK_PACKAGE_MANAGER = "conda"
SUPPORTED_PACKAGE_MANAGERS = [DEFAULT_PACKAGE_MANAGER, FALLBACK_PACKAGE_MANAGER]


_logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build queens environment on remote machine.")
    parser.add_argument(
        '--host', type=str, required=True, help='hostname or ip address of remote host'
    )
    parser.add_argument('--user', type=str, required=True, help='remote username')
    parser.add_argument(
        '--remote-python', type=str, required=True, help='path to python environment on remote host'
    )
    parser.add_argument(
        '--remote-queens-repository',
        type=str,
        required=True,
        help='path to queens repository on remote host',
    )
    parser.add_argument(
        '--gateway',
        type=str,
        required=False,
        default=None,
        help=(
            "gateway connection (proxyjump) for remote connection in json format,"
            " e.g. '{\"host\": \"user@host\"}'"
        ),
    )
    parser.add_argument(
        '--package-manager',
        type=str,
        default=DEFAULT_PACKAGE_MANAGER,
        choices=SUPPORTED_PACKAGE_MANAGERS,
        help='package manager used for the creation of the remote environment',
    )

    args = parser.parse_args(sys.argv[1:])

    remote_connection = RemoteConnection(
        host=args.host,
        user=args.user,
        remote_python=args.remote_python,
        remote_queens_repository=args.remote_queens_repository,
        gateway=args.gateway if args.gateway is None else json.loads(args.gateway),
    )
    remote_connection.open()
    remote_connection.sync_remote_repository()
    remote_connection.build_remote_environment(package_manager=args.package_manager)
    sys.exit()
