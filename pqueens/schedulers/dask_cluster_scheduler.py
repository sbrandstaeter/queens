"""Cluster scheduler for QUEENS runs."""
import atexit
import logging
import pathlib
import socket
import time

from dask.distributed import Client, SSHCluster
from dask_jobqueue import PBSCluster, SLURMCluster

from pqueens.schedulers.dask_scheduler import Scheduler
from pqueens.utils.config_directories_dask import experiment_directory
from pqueens.utils.path_utils import PATH_TO_QUEENS
from pqueens.utils.run_subprocess import run_subprocess
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
        experiment_name,
        max_jobs,
        min_jobs,
        walltime,
        num_procs,
        num_procs_post,
        num_nodes,
        queue,
        workload_manager,
        cluster_address,
        cluster_internal_address,
        cluster_user,
        cluster_python_path,
        cluster_queens_repository,
        cluster_build_environment,
    ):
        """Init method for the cluster scheduler.

        Args:
            experiment_name (str): name of QUEENS experiment
            max_jobs (int): Maximum number of active workers on the cluster
            min_jobs (int): Minimum number of active workers for the cluster
            walltime (str): Walltime for each worker job.
            num_procs (int): number of cores per job
            num_procs_post (int): number of cores per job for post-processing
            num_nodes (int): Number of cluster nodes
            queue (str): Destination queue for each worker job
            workload_manager (str): Workload manager ("pbs" or "slurm")
            cluster_address (str): address of cluster
            cluster_internal_address (str): Internal address of cluster
            cluster_user (str): cluster username
            cluster_python_path (str): Path to Python on cluster
            cluster_queens_repository (str): Path to queens repository on cluster
            cluster_build_environment (bool): Flag to decide if queens environment should be build
                                              on cluster
        """
        self.sync_remote_repository(cluster_address, cluster_user, cluster_queens_repository)
        if cluster_build_environment:
            self.build_environment(
                cluster_address, cluster_user, cluster_queens_repository, cluster_python_path
            )

        num_cores = max(num_procs, num_procs_post)
        dask_cluster_options = get_option(VALID_WORKLOAD_MANAGERS, workload_manager)
        dask_cluster_cls = dask_cluster_options['dask_cluster_cls']
        job_extra_directives = dask_cluster_options['job_extra_directives'](num_nodes, num_cores)
        job_directives_skip = dask_cluster_options['job_directives_skip']
        cluster_python_path = cluster_python_path.replace('$HOME', f'/home/{cluster_user}')

        login_cluster = SSHCluster(
            hosts=[cluster_address, cluster_address],
            remote_python=[cluster_python_path, cluster_python_path],
            connect_options={'username': cluster_user} if cluster_user else {},
        )
        login_client = Client(login_cluster)  # links to cluster login node
        atexit.register(login_client.shutdown)

        experiment_dir = login_client.submit(experiment_directory, experiment_name).result()

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

        scheduler_port = login_client.submit(self.get_port).result()
        # Start PBS Cluster on login node
        cluster_future = login_client.submit(start_cluster_on_login_node, scheduler_port)
        try:
            client = Client(address=f"{cluster_address}:{scheduler_port}", timeout=10)
            atexit.register(client.shutdown)
            client.submit(lambda: "Dummy job").result(timeout=60)
        except OSError as error:
            cluster_future.result(timeout=10)
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
        num_nodes = scheduler_options.get('num_nodes', 1)
        queue = scheduler_options.get('queue', 'batch')

        workload_manager = scheduler_options['workload_manager']
        cluster_address = scheduler_options['cluster_address']
        cluster_internal_address = scheduler_options.get('cluster_internal_address')
        cluster_user = scheduler_options.get('cluster_user')
        cluster_python_path = scheduler_options['cluster_python_path']

        cluster_queens_repository = scheduler_options.get(
            'cluster_queens_repository', f'/home/{cluster_user}/workspace/queens'
        )
        cluster_build_environment = scheduler_options.get('build_queens_environment', False)

        return cls(
            experiment_name=experiment_name,
            max_jobs=max_jobs,
            min_jobs=min_jobs,
            walltime=walltime,
            num_procs=num_procs,
            num_procs_post=num_procs_post,
            num_nodes=num_nodes,
            queue=queue,
            workload_manager=workload_manager,
            cluster_address=cluster_address,
            cluster_internal_address=cluster_internal_address,
            cluster_user=cluster_user,
            cluster_python_path=cluster_python_path,
            cluster_queens_repository=cluster_queens_repository,
            cluster_build_environment=cluster_build_environment,
        )

    @staticmethod
    def sync_remote_repository(cluster_address, cluster_user, cluster_queens_repository):
        """Synchronize local and remote QUEENS source files.

        Args:
            cluster_address (str): address of cluster
            cluster_user (str): cluster username
            cluster_queens_repository (str): Path to queens repository on cluster
        """
        if cluster_user:
            cluster_address = cluster_user + '@' + cluster_address
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
            f"{cluster_address}:{cluster_queens_repository}",
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

    @staticmethod
    def build_environment(
        cluster_address, cluster_user, cluster_queens_repository, cluster_python_path
    ):
        """Build remote QUEENS environment.

        Args:
            cluster_address (str): address of cluster
            cluster_user (str): cluster username
            cluster_queens_repository (str): Path to queens repository on cluster
            cluster_python_path (str): Path to Python on cluster
        """
        if cluster_user:
            cluster_address = cluster_user + '@' + cluster_address
        _logger.info("Build remote QUEENS environment...")
        environment_name = pathlib.Path(cluster_python_path).parents[1].name
        command_string = (
            f'ssh {cluster_address} "'
            f'cd {cluster_queens_repository}; '
            f'conda env create -f environment.yml --name {environment_name} --force; '
            f'conda activate {environment_name}; which conda; '
            f'pip install -e ."'
        )
        start_time = time.time()
        _, _, stdout, _ = run_subprocess(
            command_string,
            raise_error_on_subprocess_failure=False,
        )
        _logger.debug(stdout)
        _logger.info("Build of remote queens environment was successful.")
        _logger.info("It took: %s s.\n", time.time() - start_time)

    @staticmethod
    def get_port():
        """Get free port.

        Returns:
            int: free port
        """
        sock = socket.socket()
        sock.bind(('', 0))
        return sock.getsockname()[1]
