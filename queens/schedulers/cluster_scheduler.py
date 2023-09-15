"""Cluster scheduler for QUEENS runs."""
import logging
import socket
import subprocess
import time
from datetime import timedelta

from dask.distributed import Client
from dask_jobqueue import PBSCluster, SLURMCluster

import queens.global_settings
from queens.schedulers.scheduler import Scheduler
from queens.utils.valid_options_utils import get_option

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


def timedelta_to_str(timedelta_obj):
    """Format a timedelta object to str.

    This function seems unnecessarily complicated, but unfortunately the datetime library does not
     support this formatting for timedeltas. Returns the format HH:MM:SS.

    Args:
        timedelta_obj (datetime.timedelta): Timedelta object to format

    Returns:
        str: String of the timedelta object
    """
    # Time in seconds
    time_in_seconds = int(timedelta_obj.total_seconds())
    (minutes, seconds) = divmod(time_in_seconds, 60)
    (hours, minutes) = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


class ClusterScheduler(Scheduler):
    """Cluster scheduler for QUEENS."""

    def __init__(
        self,
        workload_manager,
        walltime,
        max_jobs=1,
        min_jobs=0,
        num_procs=1,
        num_procs_post=1,
        num_nodes=1,
        queue=None,
        cluster_internal_address=None,
        restart_workers=True,
        allowed_failures=5,
    ):
        """Init method for the cluster scheduler.

        The total number of cores per job is given by num_procs*num_nodes.

        Args:
            workload_manager (str): Workload manager ("pbs" or "slurm")
            walltime (str): Walltime for each worker job. Format (hh:mm:ss)
            max_jobs (int, opt): Maximum number of active workers on the cluster
            min_jobs (int, opt): Minimum number of active workers for the cluster
            num_procs (int, opt): Number of cores per job per node
            num_procs_post (int, opt): Number of cores per job for post-processing
            num_nodes (int, opt): Number of cluster nodes per job
            queue (str, opt): Destination queue for each worker job
            cluster_internal_address (str, opt): Internal address of cluster
            restart_workers (bool): If true, restart workers after each finished job. For larger
                                    jobs (>1min) this should be set to true in most cases.
            allowed_failures (int): Number of allowed failures for a task before an error is raised
        """
        experiment_name = queens.global_settings.GLOBAL_SETTINGS.experiment_name

        num_cores = max(num_procs, num_procs_post)
        dask_cluster_options = get_option(VALID_WORKLOAD_MANAGERS, workload_manager)
        job_extra_directives = dask_cluster_options['job_extra_directives'](num_nodes, num_cores)
        job_directives_skip = dask_cluster_options['job_directives_skip']
        if queue is None:
            job_directives_skip.append('#SBATCH -p')

        hours, minutes, seconds = map(int, walltime.split(':'))
        walltime_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)

        # Increase jobqueue walltime by 5 minutes to kill dask workers in time
        walltime = timedelta_to_str(walltime_delta + timedelta(minutes=5))

        # dask worker lifetime = walltime - 3m +/- 2m
        worker_lifetime = str(int((walltime_delta + timedelta(minutes=2)).total_seconds())) + "s"

        remote_port = queens.global_settings.GLOBAL_SETTINGS.remote_port
        remote_port_dashboard = queens.global_settings.GLOBAL_SETTINGS.remote_port_dashboard
        scheduler_options = {
            "port": remote_port,
            "dashboard_address": remote_port_dashboard,
            "allowed_failures": allowed_failures,
        }
        if cluster_internal_address:
            scheduler_options["contact_address"] = f"{cluster_internal_address}:{remote_port}"
        dask_cluster_kwargs = {
            "job_name": experiment_name,
            "queue": queue,
            "cores": num_cores,
            "memory": '10TB',
            "scheduler_options": scheduler_options,
            "walltime": walltime,
            "log_directory": str(queens.global_settings.GLOBAL_SETTINGS.remote_experiment_dir),
            "job_directives_skip": job_directives_skip,
            "job_extra_directives": [job_extra_directives],
            "worker_extra_args": ["--lifetime", worker_lifetime, "--lifetime-stagger", "2m"],
        }
        dask_cluster_adapt_kwargs = {
            "minimum_jobs": min_jobs,
            "maximum_jobs": max_jobs,
        }
        stdout, stderr = queens.global_settings.GLOBAL_SETTINGS.remote_connection.start_cluster(
            workload_manager,
            dask_cluster_kwargs,
            dask_cluster_adapt_kwargs,
            queens.global_settings.GLOBAL_SETTINGS.remote_experiment_dir,
        )
        _logger.debug(stdout)
        _logger.debug(stderr)

        local_port = queens.global_settings.GLOBAL_SETTINGS.local_port
        local_port_dashboard = queens.global_settings.GLOBAL_SETTINGS.local_port_dashboard

        # connection.open_port_forwarding(local_port, remote_port)
        # connection.open_port_forwarding(local_port_dashboard, remote_port_dashboard)
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
            experiment_dir=queens.global_settings.GLOBAL_SETTINGS.remote_experiment_dir,
            client=client,
            num_procs=num_procs,
            num_procs_post=num_procs_post,
            restart_workers=restart_workers,
        )

    def restart_worker(self, worker):
        """Restart a worker.

        This method cancels the job in the queue of the HPC system. The Client.adapt method of dask
        will subsequently submit new jobs to the queue. Warning: Currently only slurm is supported.

        Args:
            worker (str, tuple): Worker to restart. This can be a worker address, name, or a both.
        """
        job_id = self.client.run(
            lambda: subprocess.check_output('echo $SLURM_JOB_ID', shell=True),
            workers=list(worker),
        )
        cancel_cmd = f'scancel {str(list(job_id.values())[0])[2:-3]}'
        self.client.run_on_scheduler(lambda: subprocess.run(cancel_cmd, check=False, shell=True))

    @staticmethod
    def get_port():
        """Get free port.

        Returns:
            int: free port
        """
        sock = socket.socket()
        sock.bind(('', 0))
        return sock.getsockname()[1]
