"""Local scheduler for QUEENS runs."""

import logging

from dask.distributed import Client, LocalCluster

from queens.schedulers.scheduler import Scheduler
from queens.utils.config_directories import experiment_directory
from queens.utils.logger_settings import log_init_args
from queens.utils.rsync import rsync

_logger = logging.getLogger(__name__)


class LocalScheduler(Scheduler):
    """Local scheduler class for QUEENS."""

    @log_init_args
    def __init__(
        self,
        experiment_name,
        max_concurrent=1,
        num_procs=1,
        restart_workers=False,
    ):
        """Initialize local scheduler.

        Args:
            experiment_name (str): name of the current experiment
            max_concurrent (int, opt): Number of concurrent jobs
            num_procs (int, opt): number of processors per job
            restart_workers (bool): If true, restart workers after each finished job. Try setting it
                                    to true in case you are experiencing memory-leakage warnings.
        """
        experiment_dir = experiment_directory(experiment_name=experiment_name)

        cluster = LocalCluster(
            n_workers=max_concurrent,
            processes=True,
            threads_per_worker=num_procs,
            silence_logs=False,
        )
        client = Client(cluster)
        _logger.info(
            "To view the Dask dashboard open this link in your browser: %s", client.dashboard_link
        )
        super().__init__(
            experiment_name=experiment_name,
            experiment_dir=experiment_dir,
            client=client,
            num_procs=num_procs,
            restart_workers=restart_workers,
        )

    def restart_worker(self, worker):
        """Restart a worker.

        Args:
            worker (str, tuple): Worker to restart. This can be a worker address, name, or a both.
        """
        self.client.restart_workers(workers=list(worker))

    def copy_files_to_experiment_dir(self, paths):
        """Copy file to experiment directory.

        Args:
            paths (Path, list): paths to files or directories that should be copied to experiment
                                directory
        """
        destination = f"{self.experiment_dir}/"
        rsync(paths, destination)
