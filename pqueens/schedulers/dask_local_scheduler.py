"""Local scheduler for QUEENS runs."""
import logging

from dask.distributed import Client, LocalCluster

from pqueens.schedulers.dask_scheduler import Scheduler
from pqueens.utils.config_directories_dask import experiment_directory

_logger = logging.getLogger(__name__)


class LocalScheduler(Scheduler):
    """Local scheduler class for QUEENS."""

    @classmethod
    def from_config_create_scheduler(cls, config, scheduler_name):
        """Create standard scheduler object from config.

        Args:
            config (dict): QUEENS input dictionary
            scheduler_name (str): Name of the scheduler

        Returns:
            Instance of  LocalScheduler class
        """
        scheduler_options = config[scheduler_name]
        experiment_name = config['global_settings']['experiment_name']
        experiment_dir = experiment_directory(experiment_name=experiment_name)

        max_concurrent = scheduler_options.get('max_concurrent', 1)
        num_procs = scheduler_options.get('num_procs', 1)
        num_procs_post = scheduler_options.get('num_procs_post', 1)
        threads_per_worker = max(num_procs, num_procs_post)

        cluster = LocalCluster(
            n_workers=max_concurrent,
            processes=False,
            threads_per_worker=threads_per_worker,
        )
        client = Client(cluster)

        return cls(experiment_name, experiment_dir, client, num_procs, num_procs_post)
