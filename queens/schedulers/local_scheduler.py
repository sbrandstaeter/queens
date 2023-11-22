"""Local scheduler for QUEENS runs."""
import logging

from dask.distributed import Client, LocalCluster

import queens.global_settings
from queens.schedulers.scheduler import Scheduler
from queens.utils.config_directories import experiment_directory
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class LocalScheduler(Scheduler):
    """Local scheduler class for QUEENS."""

    @log_init_args
    def __init__(self, max_concurrent=1, num_procs=1, num_procs_post=1, restart_workers=False):
        """Initialize local scheduler.

        Args:
            max_concurrent (int, opt): Number of concurrent jobs
            num_procs (int, opt): number of cores per job
            num_procs_post (int, opt): number of cores per job for post-processing
            restart_workers (bool): If true, restart workers after each finished job. Try setting it
                                    to true in case you are experiencing memory-leakage warnings.
        """
        experiment_name = queens.global_settings.GLOBAL_SETTINGS.experiment_name
        experiment_dir = experiment_directory(experiment_name=experiment_name)

        threads_per_worker = max(num_procs, num_procs_post)
        cluster = LocalCluster(
            n_workers=max_concurrent,
            processes=True,
            threads_per_worker=threads_per_worker,
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
            num_procs_post=num_procs_post,
            restart_workers=restart_workers,
        )

    def restart_worker(self, worker):
        """Restart a worker.

        Args:
            worker (str, tuple): Worker to restart. This can be a worker address, name, or a both.
        """
        self.client.restart_workers(workers=list(worker))
