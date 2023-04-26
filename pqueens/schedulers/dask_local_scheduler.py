"""Local scheduler for QUEENS runs."""
import atexit
import logging

from dask.distributed import Client, LocalCluster

from pqueens.schedulers.dask_scheduler import Scheduler
from pqueens.utils.config_directories_dask import experiment_directory

_logger = logging.getLogger(__name__)


class LocalScheduler(Scheduler):
    """Local scheduler class for QUEENS."""

    def __init__(
        self,
        global_settings,
        max_concurrent=1,
        num_procs=1,
        num_procs_post=1,
    ):
        """Initialize local scheduler.

        Args:
            global_settings (dict): Dictionary containing global settings for the QUEENS run.
            max_concurrent (int, opt): Number of concurrent jobs
            num_procs (int, opt): number of cores per job
            num_procs_post (int, opt): number of cores per job for post-processing
        """
        experiment_name = global_settings['experiment_name']
        experiment_dir = experiment_directory(experiment_name=experiment_name)

        threads_per_worker = max(num_procs, num_procs_post)
        cluster = LocalCluster(
            n_workers=max_concurrent,
            processes=False,
            threads_per_worker=threads_per_worker,
            silence_logs=False,
        )
        client = Client(cluster)
        atexit.register(client.shutdown)
        super().__init__(experiment_name, experiment_dir, client, num_procs, num_procs_post)
