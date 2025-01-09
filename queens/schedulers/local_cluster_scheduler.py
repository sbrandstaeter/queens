"""Cluster scheduler for QUEENS runs."""

import logging
import time
from datetime import timedelta

from dask.distributed import Client

from queens.schedulers.cluster_scheduler import VALID_WORKLOAD_MANAGERS, timedelta_to_str
from queens.schedulers.dask_scheduler import DaskScheduler
from queens.utils.config_directories import experiment_directory  # Do not change this import!
from queens.utils.logger_settings import log_init_args
from queens.utils.remote_operations import get_port
from queens.utils.rsync import rsync
from queens.utils.valid_options_utils import get_option

_logger = logging.getLogger(__name__)


class LocalClusterScheduler(DaskScheduler):
    """Local Cluster scheduler for QUEENS.

    Can be used to schedule jobs to a cluster scheduler with local
    access i.e. without a network connection.
    """

    @log_init_args
    def __init__(
        self,
        experiment_name,
        workload_manager,
        walltime,
        num_jobs=1,
        min_jobs=0,
        num_procs=1,
        num_nodes=1,
        queue=None,
        cluster_internal_address=None,
        restart_workers=False,
        allowed_failures=5,
    ):
        """Init method for the cluster scheduler.

        The total number of cores per job is given by num_procs*num_nodes.

        Args:
            experiment_name (str): name of the current experiment
            workload_manager (str): Workload manager ("pbs" or "slurm")
            walltime (str): Walltime for each worker job. Format (hh:mm:ss)
            num_jobs (int, opt): Maximum number of parallel jobs
            min_jobs (int, opt): Minimum number of active workers for the cluster
            num_procs (int, opt): Number of processors per job per node
            num_nodes (int, opt): Number of cluster nodes per job
            queue (str, opt): Destination queue for each worker job
            cluster_internal_address (str, opt): Internal address of cluster
            restart_workers (bool): If true, restart workers after each finished job. For larger
                                    jobs (>1min) this should be set to true in most cases.
            allowed_failures (int): Number of allowed failures for a task before an error is raised
        """
        experiment_dir = experiment_directory(experiment_name=experiment_name)
        _logger.debug(
            "experiment directory: %s",
            experiment_dir,
        )

        # collect all settings for the dask cluster
        dask_cluster_options = get_option(VALID_WORKLOAD_MANAGERS, workload_manager)
        job_extra_directives = dask_cluster_options["job_extra_directives"](num_nodes, num_procs)
        job_directives_skip = dask_cluster_options["job_directives_skip"]
        if queue is None:
            job_directives_skip.append("#SBATCH -p")

        hours, minutes, seconds = map(int, walltime.split(":"))
        walltime_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)

        # Increase jobqueue walltime by 5 minutes to kill dask workers in time
        walltime = timedelta_to_str(walltime_delta + timedelta(minutes=5))

        # dask worker lifetime = walltime - 3m +/- 2m
        worker_lifetime = str(int((walltime_delta + timedelta(minutes=2)).total_seconds())) + "s"

        remote_port = get_port()
        local_port_dashboard = get_port()
        remote_port_dashboard = get_port()

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
            "memory": "10TB",
            "scheduler_options": scheduler_options,
            "walltime": walltime,
            "log_directory": str(experiment_dir),
            "job_directives_skip": job_directives_skip,
            "job_extra_directives": [job_extra_directives],
            "worker_extra_args": ["--lifetime", worker_lifetime, "--lifetime-stagger", "2m"],
            # keep this hardcoded to 1, the number of threads for the mpi run is handled by
            # job_extra_directives. Note that the number of workers is not the number of parallel
            # simulations!
            "cores": 1,
            "processes": 1,
            "n_workers": 1,
        }
        dask_cluster_adapt_kwargs = {
            "minimum_jobs": min_jobs,
            "maximum_jobs": num_jobs,
        }

        dask_cluster_options = get_option(VALID_WORKLOAD_MANAGERS, workload_manager)
        dask_cluster_cls = dask_cluster_options["dask_cluster_cls"]

        try:
            _logger.info("Starting dask cluster of type: %s", dask_cluster_cls)
            _logger.debug("Dask cluster kwargs:")
            _logger.debug(dask_cluster_kwargs)
            cluster = dask_cluster_cls(**dask_cluster_kwargs)

            _logger.info("Adapting dask cluster settings")
            _logger.debug("Dask cluster adapt kwargs:")
            _logger.debug(dask_cluster_adapt_kwargs)
            cluster.adapt(**dask_cluster_adapt_kwargs)

            _logger.info("Dask cluster info:")
            _logger.info(cluster)

            dask_jobscript = experiment_dir / "dask_jobscript.sh"
            _logger.info("Writing dask jobscript to:")
            _logger.info(dask_jobscript)
            dask_jobscript.write_text(str(cluster.job_script()))
        except Exception as e:
            raise RuntimeError() from e

        for i in range(20, 0, -1):  # 20 tries to connect
            _logger.debug("Trying to connect to Dask Cluster: try #%d", i)
            try:
                # client = Client(address=f"localhost:{local_port}", timeout=10)
                client = Client(cluster)
                break
            except OSError as exc:
                if i == 1:
                    raise OSError() from exc
                time.sleep(1)

        _logger.debug("Submitting dummy job to check basic functionality of client.")
        client.submit(lambda: "Dummy job").result(timeout=180)
        _logger.debug("Dummy job was successful.")
        _logger.info(
            "To view the Dask dashboard open this link in your browser: "
            "http://localhost:%i/status",
            local_port_dashboard,
        )

        super().__init__(
            experiment_name=experiment_name,
            experiment_dir=experiment_dir,
            num_jobs=num_jobs,
            num_procs=num_procs,
            client=client,
            restart_workers=restart_workers,
        )

    def restart_worker(self, worker):
        """Restart a worker.

        This method retires a dask worker. The Client.adapt method of dask takes cares of submitting
        new workers subsequently.

        Args:
            worker (str, tuple): Worker to restart. This can be a worker address, name, or a both.
        """
        self.client.retire_workers(workers=list(worker))

    def copy_files_to_experiment_dir(self, paths):
        """Copy file to experiment directory.

        Args:
            paths (Path, list): paths to files or directories that should be copied to experiment
                                directory
        """
        destination = f"{self.experiment_dir}/"
        rsync(paths, destination)
