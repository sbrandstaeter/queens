"""Cluster scheduler for QUEENS runs."""
import atexit
import logging
import socket
import time
from datetime import datetime, timedelta

from dask.distributed import Client
from dask_jobqueue import PBSCluster, SLURMCluster

import pqueens.global_settings
from pqueens.schedulers.scheduler import Scheduler
from pqueens.utils.config_directories import experiment_directory
from pqueens.utils.remote_build import build_remote_environment, sync_remote_repository
from pqueens.utils.remote_operations import RemoteConnection
from pqueens.utils.valid_options_utils import get_option

_logger = logging.getLogger(__name__)

VALID_WORKLOAD_MANAGERS = {
    "slurm": {
        "dask_cluster_cls": SLURMCluster,
        "job_extra_directives": lambda nodes, cores: f"--ntasks={nodes * cores}",
        "job_directives_skip": [
            '#SBATCH -n 1',
            '#SBATCH --mem=',
            '#SBATCH --cpus-per-task=',
        ],
    },
    "pbs": {
        "dask_cluster_cls": PBSCluster,
        "job_extra_directives": lambda nodes, cores: f"-l nodes={nodes}:ppn={cores}",
        "job_directives_skip": ["#PBS -l select"],
    },
}


class ClusterScheduler(Scheduler):
    """Cluster scheduler for QUEENS."""

    def __init__(
        self,
        workload_manager,
        cluster_address,
        cluster_user,
        cluster_python_path,
        walltime,
        max_jobs=1,
        min_jobs=0,
        num_procs=1,
        num_procs_post=1,
        num_nodes=1,
        queue=None,
        cluster_internal_address=None,
        cluster_queens_repository=None,
        cluster_build_environment=False,
        progressbar=True,
    ):
        """Init method for the cluster scheduler.

        Args:
            workload_manager (str): Workload manager ("pbs" or "slurm")
            cluster_address (str): address of cluster
            cluster_user (str): cluster username
            cluster_python_path (str): Path to Python on cluster
            walltime (str): Walltime for each worker job. Format (hh:mm:ss)
            max_jobs (int, opt): Maximum number of active workers on the cluster
            min_jobs (int, opt): Minimum number of active workers for the cluster
            num_procs (int, opt): number of cores per job
            num_procs_post (int, opt): number of cores per job for post-processing
            num_nodes (int, opt): Number of cluster nodes
            queue (str, opt): Destination queue for each worker job
            cluster_internal_address (str, opt): Internal address of cluster
            cluster_queens_repository (str, opt): Path to Queens repository on cluster
            cluster_build_environment (bool, opt): Flag to decide if queens environment should be
                                                   build on cluster
            progressbar (bool, opt): If true, print progressbar. WARNING: If multiple dask
                                     schedulers are used, the progressbar must be disabled.
        """
        if cluster_queens_repository is None:
            cluster_queens_repository = f'/home/{cluster_user}/workspace/queens'
        _logger.debug("cluster queens repository: %s", cluster_queens_repository)

        experiment_name = pqueens.global_settings.GLOBAL_SETTINGS.experiment_name

        sync_remote_repository(cluster_address, cluster_user, cluster_queens_repository)

        _logger.debug("cluster python path: %s", cluster_python_path)
        if cluster_build_environment:
            build_remote_environment(
                cluster_address, cluster_user, cluster_queens_repository, cluster_python_path
            )

        num_cores = max(num_procs, num_procs_post)
        dask_cluster_options = get_option(VALID_WORKLOAD_MANAGERS, workload_manager)
        job_extra_directives = dask_cluster_options['job_extra_directives'](num_nodes, num_cores)
        job_directives_skip = dask_cluster_options['job_directives_skip']
        if queue is None:
            job_directives_skip.append('#SBATCH -p')

        connection = RemoteConnection(cluster_address, cluster_python_path, user=cluster_user)
        connection.open()
        atexit.register(connection.close)

        # note that we are executing the command on remote directly such the local version of
        # experiment_directory has to be used
        experiment_dir = connection.run_function(experiment_directory, experiment_name)
        _logger.debug(
            "experiment directory on %s@%s: %s", cluster_user, cluster_address, experiment_dir
        )
        walltime_delta = datetime.strptime(walltime, "%H:%M:%S") - datetime.strptime("0", "%S")
        # Increase jobqueue walltime by 5 minutes to kill dask workers in time
        walltime = str(walltime_delta + timedelta(minutes=5))
        # dask worker lifetime = walltime - 3m +/- 2m
        worker_lifetime = str((walltime_delta + timedelta(minutes=2)).seconds) + "s"

        remote_port = connection.run_function(self.get_port)
        remote_port_dashboard = connection.run_function(self.get_port)
        scheduler_options = {"port": remote_port, "dashboard_address": remote_port_dashboard}
        if cluster_internal_address:
            scheduler_options["contact_address"] = f"{cluster_internal_address}:{remote_port}"
        dask_cluster_kwargs = {
            "job_name": experiment_name,
            "queue": queue,
            "cores": num_cores,
            "memory": '10TB',
            "scheduler_options": scheduler_options,
            "walltime": walltime,
            "log_directory": str(experiment_dir),
            "job_directives_skip": job_directives_skip,
            "job_extra_directives": [job_extra_directives],
            "worker_extra_args": ["--lifetime", worker_lifetime, "--lifetime-stagger", "2m"],
        }
        dask_cluster_adapt_kwargs = {
            "minimum_jobs": min_jobs,
            "maximum_jobs": max_jobs,
        }
        stdout, stderr = connection.start_cluster(
            cluster_queens_repository,
            workload_manager,
            dask_cluster_kwargs,
            dask_cluster_adapt_kwargs,
            experiment_dir,
        )

        local_port = self.get_port()
        local_port_dashboard = self.get_port()

        connection.open_port_forwarding(local_port, remote_port)
        connection.open_port_forwarding(local_port_dashboard, remote_port_dashboard)
        for i in range(20, 0, -1):  # 20 tries to connect
            _logger.debug("Trying to connect to Dask Cluster: try #%d", i)
            try:
                client = Client(address=f"localhost:{local_port}", timeout=10)
                break
            except OSError as exc:
                if i == 1:
                    raise OSError(
                        stdout.read().decode('ascii') + stderr.read().decode('ascii')
                    ) from exc
                time.sleep(1)

        _logger.debug("Submitting dummy job to check basic functionality of client.")
        client.submit(lambda: "Dummy job").result(timeout=180)
        _logger.debug("Dummy job was successful.")
        _logger.info(
            'To view the Dask dashboard open this link in your browser: '
            'http://localhost:%i/status',
            local_port_dashboard,
        )

        super().__init__(
            experiment_name=experiment_name,
            experiment_dir=experiment_dir,
            client=client,
            num_procs=num_procs,
            num_procs_post=num_procs_post,
            progressbar=progressbar,
        )

    @staticmethod
    def get_port():
        """Get free port.

        Returns:
            int: free port
        """
        sock = socket.socket()
        sock.bind(('', 0))
        return sock.getsockname()[1]
