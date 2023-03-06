"""Cluster scheduler for QUEENS runs."""
import logging
import time

from dask.distributed import Client, SSHCluster
from dask_jobqueue import PBSCluster, SLURMCluster

from pqueens.schedulers.dask_scheduler import Scheduler
from pqueens.utils import config_directories_dask
from pqueens.utils.config_directories_dask import experiment_directory, remote_queens_directory
from pqueens.utils.path_utils import PATH_TO_QUEENS
from pqueens.utils.run_subprocess import run_subprocess
from pqueens.utils.valid_options_utils import get_option

_logger = logging.getLogger(__name__)

VALID_WORKLOAD_MANAGERS = {"slurm": SLURMCluster, "pbs": PBSCluster}


class ClusterScheduler(Scheduler):
    """Cluster scheduler for QUEENS."""

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
        """Init method for the cluster scheduler.

        Args:
            experiment_name (str): name of QUEENS experiment
            max_jobs (int): Maximum number of active workers on the cluster
            min_jobs (int): Minimum number of active workers for the cluster
            walltime (str): Walltime for each worker job.
            num_procs (int): number of cores per job
            num_procs_post (int): number of cores per job for post-processing
            scheduler_port (int): Port of dask cluster scheduler
            dask_cluster_cls (obj): Dask SlurmCluster or PBSCluster class
            cluster_address (str): address of cluster
            cluster_python_path (str): Path to Python on cluster
        """
        login_cluster = SSHCluster(
            hosts=[cluster_address, cluster_address],
            remote_python=[cluster_python_path, cluster_python_path],
        )
        login_client = Client(login_cluster)  # links to cluster master node
        login_client.upload_file(config_directories_dask.__file__)

        future = login_client.submit(experiment_directory, experiment_name)
        experiment_dir = future.result()

        future = login_client.submit(remote_queens_directory)
        repository_dir = future.result()

        # TODO: should we use client.upload_file instead?
        self.sync_remote_repository(cluster_address, repository_dir)

        def start_cluster_on_login_node():
            """Start dask cluster object on login node"""
            cores = max(num_procs, num_procs_post)
            cluster = dask_cluster_cls(
                queue='batch',
                cores=cores,
                memory='10TB',
                scheduler_options={"port": scheduler_port},
                walltime=walltime,
                job_script_prologue=[f"#PBS -l nodes=1:ppn={cores}"],
            )
            cluster.adapt(minimum_jobs=min_jobs, maximum_jobs=max_jobs)
            while True:
                time.sleep(1)

        # Start PBS Cluster on master node
        cluster_future = login_client.submit(start_cluster_on_login_node)

        try:
            client = Client(address=f"{cluster_address}:{scheduler_port}")
        except OSError as error:
            cluster_future.result()
            raise error

        super().__init__(experiment_name, experiment_dir, client, num_procs, num_procs_post)

    @classmethod
    def from_config_create_scheduler(cls, config, scheduler_name):
        """Create scheduler object from config.

        Args:
            config (dict): QUEENS input dictionary
            scheduler_name (str): Name of the scheduler

        Returns:
            Instance of scheduler class
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

    @staticmethod
    def sync_remote_repository(cluster_address, repository_dir):
        """Synchronize local and remote QUEENS source files.

        Args:
            cluster_address (str): address of cluster
            repository_dir (Path): pathlib Path to remote repository
        """
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
            f"{cluster_address}:{repository_dir}",
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
