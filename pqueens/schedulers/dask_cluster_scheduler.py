"""Cluster scheduler for QUEENS runs."""
import logging
import time

from dask.distributed import Client, SSHCluster
from dask_jobqueue import PBSCluster, SLURMCluster

from pqueens.schedulers.dask_scheduler import Scheduler
from pqueens.utils.config_directories_dask import experiment_directory
from pqueens.utils.valid_options_utils import get_option

_logger = logging.getLogger(__name__)

VALID_WORKLOAD_MANAGERS = {"slurm": SLURMCluster, "pbs": PBSCluster}


class ClusterScheduler(Scheduler):
    def __init__(
        self,
        experiment_name,
        max_jobs,
        min_jobs,
        walltime,
        num_procs,
        num_procs_post,
        scheduler_port,
        dask_cluster_cls,
        cluster_address,
        cluster_python_path,
    ):
        login_cluster = SSHCluster(
            hosts=[cluster_address, cluster_address],
            remote_python=[cluster_python_path, cluster_python_path],
        )
        self.login_client = Client(login_cluster)  # links to cluster master node

        future = self.login_client.submit(experiment_directory, experiment_name)
        experiment_dir = future.result()

        def start_cluster_on_master_node():
            cluster = dask_cluster_cls(
                queue='batch',
                cores=max(num_procs, num_procs_post),
                memory='24GB',
                scheduler_options={"port": scheduler_port},
                walltime=walltime,
            )
            cluster.adapt(minimum_jobs=min_jobs, maximum_jobs=max_jobs)
            while True:
                time.sleep(1)

        # Start PBS Cluster on master node and run as long as future object (self.cluster_future) exists
        self.cluster_future = self.login_client.submit(start_cluster_on_master_node)
        time.sleep(1)

        client = Client(address=f"{cluster_address}:{scheduler_port}")

        super().__init__(experiment_name, experiment_dir, client, num_procs, num_procs_post)

    @classmethod
    def from_config_create_scheduler(cls, config, scheduler_name):
        """Create standard scheduler object from config.

        Args:
            config (dict): QUEENS input dictionary
            scheduler_name (str): Name of the scheduler

        Returns:
            Instance of standard scheduler class
        """
        scheduler_options = config[scheduler_name]
        experiment_name = config['global_settings']['experiment_name']

        max_jobs = scheduler_options.get('max_jobs', 1)
        min_jobs = scheduler_options.get('min_jobs', 0)
        walltime = scheduler_options['walltime']

        num_procs = scheduler_options.get('num_procs', 1)
        num_procs_post = scheduler_options.get('num_procs_post', 1)

        scheduler_port = scheduler_options['scheduler_port']
        workload_manager = scheduler_options['workload_manager']
        dask_cluster_cls = get_option(VALID_WORKLOAD_MANAGERS, workload_manager)
        cluster_address = scheduler_options['cluster_address']
        cluster_python_path = scheduler_options['cluster_python_path']

        return cls(
            experiment_name,
            max_jobs,
            min_jobs,
            walltime,
            num_procs,
            num_procs_post,
            scheduler_port,
            dask_cluster_cls,
            cluster_address,
            cluster_python_path,
        )
