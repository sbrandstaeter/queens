"""Cluster scheduler for QUEENS runs."""
import atexit
import logging
import socket
import time

from dask.distributed import Client
from dask_jobqueue import PBSCluster, SLURMCluster

from pqueens.schedulers.dask_scheduler import Scheduler
from pqueens.utils.config_directories_dask import experiment_directory
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
            '#SBATCH -p batch',
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
        global_settings,
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
        queue='batch',
        cluster_internal_address=None,
        cluster_queens_repository=None,
        cluster_build_environment=False,
    ):
        """Init method for the cluster scheduler.

        Args:
            global_settings (dict): Dictionary containing global settings for the QUEENS run.
            workload_manager (str): Workload manager ("pbs" or "slurm")
            cluster_address (str): address of cluster
            cluster_user (str): cluster username
            cluster_python_path (str): Path to Python on cluster
            walltime (str): Walltime for each worker job.
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
        """
        if cluster_queens_repository is None:
            cluster_queens_repository = f'/home/{cluster_user}/workspace/queens'
        experiment_name = global_settings['experiment_name']

        sync_remote_repository(cluster_address, cluster_user, cluster_queens_repository)
        if cluster_build_environment:
            build_remote_environment(
                cluster_address, cluster_user, cluster_queens_repository, cluster_python_path
            )

        num_cores = max(num_procs, num_procs_post)
        dask_cluster_options = get_option(VALID_WORKLOAD_MANAGERS, workload_manager)
        dask_cluster_cls = dask_cluster_options['dask_cluster_cls']
        job_extra_directives = dask_cluster_options['job_extra_directives'](num_nodes, num_cores)
        job_directives_skip = dask_cluster_options['job_directives_skip']

        connection = RemoteConnection(cluster_address, cluster_python_path, user=cluster_user)
        connection.open()
        atexit.register(connection.close)

        experiment_dir = connection.run_function(experiment_directory, experiment_name)

        def start_cluster_on_login_node(port):
            """Start dask cluster object on login node."""
            scheduler_options = {"port": port}
            if cluster_internal_address:
                scheduler_options["contact_address"] = f"{cluster_internal_address}:{port}"
            cluster = dask_cluster_cls(
                job_name=experiment_name,
                queue=queue,
                cores=num_cores,
                memory='10TB',
                scheduler_options=scheduler_options,
                walltime=walltime,
                log_directory=str(experiment_dir),
                job_directives_skip=job_directives_skip,
                job_extra_directives=[job_extra_directives],
            )
            cluster.adapt(minimum_jobs=min_jobs, maximum_jobs=max_jobs)
            (experiment_dir / 'dask_jobscript').write_text(str(cluster.job_script()))
            while True:
                time.sleep(1)

        remote_port = connection.run_function(self.get_port)
        stdout, stderr = connection.run_function(
            start_cluster_on_login_node, remote_port, wait=False
        )
        local_port = self.get_port()

        connection.open_port_forwarding(local_port=local_port, remote_port=remote_port)
        for i in range(20, 0, -1):  # 20 tries to connect
            try:
                client = Client(address=f"localhost:{local_port}", timeout=10)
                break
            except OSError as exc:
                if i == 1:
                    raise OSError(
                        stdout.read().decode('ascii') + stderr.read().decode('ascii')
                    ) from exc
                time.sleep(1)
        _logger.debug(stdout.read().decode('ascii') + stderr.read().decode('ascii'))
        client.submit(lambda: "Dummy job").result(timeout=60)  # Dummy job to check functionality

        super().__init__(experiment_name, experiment_dir, client, num_procs, num_procs_post)

    @staticmethod
    def get_port():
        """Get free port.

        Returns:
            int: free port
        """
        sock = socket.socket()
        sock.bind(('', 0))
        return sock.getsockname()[1]
