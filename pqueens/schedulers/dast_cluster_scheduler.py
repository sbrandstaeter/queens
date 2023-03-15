"""Package to manage dask-jobqueue clusters."""

import logging
import time

from pqueens.utils.config_directories import remote_queens_directory
from pqueens.utils.path_utils import PATH_TO_QUEENS
from pqueens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)


class RemoteClusterManager:
    """Manage remote jobqueue cluster from local machine.

    Attributes:
        cluster:
    """

    def __init__(
        self,
        remote_connect,
        scheduler_port,
        scheduler_address,
        cores_per_worker,
        num_workers,
        dashboard_port,
    ):
        """Create a RemoteClusterManager."""
        self.remote_connect = remote_connect

        self.cluster_port_forwarding_command = (
            f"ssh -f -N -L "
            f"{scheduler_port}:{scheduler_address}:{scheduler_port} "
            f"{self.remote_connect}"
        )
        self.dashboard_port_forwarding_command = (
            f"ssh -f -N -L "
            f"{dashboard_port}:{scheduler_address}:{dashboard_port} "
            f"{self.remote_connect}"
        )

        self.remote_queens_directory = remote_queens_directory(remote_connect=remote_connect)

        slurm_cluster = self.remote_queens_directory / 'pqueens' / 'cluster' / 'slurm_cluster.py'
        self.run_cluster_command = (
            f"python -u {slurm_cluster} "
            f"--scheduler-port {scheduler_port} "
            f"--dashboard-port {dashboard_port} "
            f"--cores-per-worker {cores_per_worker} "
            f"--num-workers {num_workers}"
        )

    def __enter__(self):
        """'enter'-function for usage as a context.

        This function is called
        prior to entering the context
        It is used to:
            1. synchronize remote and local data
            1. start the cluster
            2. establish the port-forwarding from remote to local

        Returns:
            self
        """
        self.setup_cluster()

    def __exit__(self, exception_type, exception_value, traceback):
        """'exit'-function in order to use the db objects as a context.

        This function is called at the end of the context in order to close the connection to the
        database.

        The exception as well as traceback arguments are required to implement the `__exit__`
        method, however, we do not use them explicitly.

        Args:
            exception_type: indicates class of exception (e.g. ValueError)
            exception_value: indicates exception instance
            traceback: traceback object
        """
        self.stop_remote_cluster()
        self.close_port_forwarding()
        if exception_type:
            _logger.exception(exception_type(exception_value).with_traceback(traceback))

    def setup_cluster(self):
        """Collect all methods needed to set up the cluster."""
        self.sync_remote_repository()
        self.sync_remote_environment()
        self.start_remote_cluster()
        self.open_port_forwarding()
        # allow some time to properly start the cluster
        time.sleep(5)

    def shutdown_cluster(self):
        """Collect all methods to properly shut cluster down."""
        self.stop_remote_cluster()
        self.close_port_forwarding()

    def sync_remote_repository(self):
        """Synchronize local and remote QUEENS source files."""
        _logger.info("Syncing remote QUEENS repository with local one...")
        command_list = [
            "rsync --archive --checksum --verbose --verbose",
            "--exclude '.git'",
            "--exclude '.eggs'",
            "--exclude '.gitlab'",
            "--exclude '.idea'",
            "--exclude '.vscode'",
            "--exclude '.pytest_cache'",
            "--exclude '__pycache__'",
            "--exclude 'doc'",
            "--exclude 'html_coverage_report'",
            "--exclude 'config'",
            f"{PATH_TO_QUEENS}/",
            f"{self.remote_connect}:{self.remote_queens_directory}",
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

    def sync_remote_environment(self):
        """Synchronize local and remote Python environment."""

    def start_remote_cluster(self):
        """Start remote cluster."""
        command_list = [
            f"ssh {self.remote_connect}",
            "'",
            "conda activate queens-remote;",
            "nohup",
            self.run_cluster_command,
            "&",
            "'",
        ]
        command_string = ' '.join(command_list)
        _, process_id, stdout, _ = run_subprocess(
            command_string,
            subprocess_type="submit",
            additional_error_message="Error during start of remote cluster! ",
        )
        _logger.debug(process_id)
        _logger.debug(stdout)
        _logger.info("Cluster started successfully.\n")

    def stop_remote_cluster(self):
        """Stop remote cluster."""
        command_list = [
            f"ssh {self.remote_connect}",
            "'",
            "pkill -f",
            '"' + self.run_cluster_command + '"',
            "'",
        ]
        command_string = ' '.join(command_list)
        _, process_id, stdout, _ = run_subprocess(
            command_string,
            additional_error_message="Error during stopping of remote cluster! ",
        )
        _logger.debug(process_id)
        _logger.debug(stdout)
        _logger.info("Cluster stopped successfully.\n")

    def open_port_forwarding(self):
        """Establish port forwarding from remote to local."""
        _, process_id, stdout, _ = run_subprocess(
            self.cluster_port_forwarding_command,
            subprocess_type="submit",
            additional_error_message="Error during opening port-forwarding to cluster! ",
        )
        _logger.debug(process_id)
        _logger.debug(stdout)

        _logger.info("Port-forwarding to cluster opened successfully.")
        _logger.info("To open a port-forwarding for the dashboard use:")
        _logger.info(self.dashboard_port_forwarding_command)
        _logger.info("\n")

    def close_port_forwarding(self):
        """Close port forwarding from remote to local."""
        command_list = [
            "pkill -f",
            '"' + self.cluster_port_forwarding_command + '"',
        ]
        command_string = ' '.join(command_list)
        _, process_id, stdout, _ = run_subprocess(
            command_string,
            additional_error_message="Error during closing port-forwarding to cluster! ",
        )
        _logger.debug(process_id)
        _logger.debug(stdout)

        _logger.info("Port-forwarding closed successfully.\n")
