"""Driver to run an executable with mpi."""

import logging
import pathlib

import pqueens.database.database as DB_module
from pqueens.data_processor import from_config_create_data_processor
from pqueens.drivers.driver import Driver
from pqueens.schedulers.cluster_scheduler import (
    VALID_CLUSTER_SCHEDULER_TYPES,
    VALID_PBS_SCHEDULER_TYPES,
)
from pqueens.utils.cluster_utils import get_cluster_job_id
from pqueens.utils.injector import inject
from pqueens.utils.print_utils import get_str_table
from pqueens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)


class MpiDriver(Driver):
    """Driver to run an executable with mpi.

    Attributes:
        cae_output_streaming (bool): flag for additional streaming to given stream
        cluster_job (bool): true if job is executed on a cluster
        cluster_options (dict): cluster options for pbs or slurm
        error_file (path): path to error file
        executable (path): path to main executable of respective software (e.g. baci)
        input_file (path): path to input file
        log_file (path): path to log file
        mpi_cmd (str): mpi command
        num_procs_post (int): number of processors for post-processing
        output_file (path): path to output file
        output_prefix (str): output prefix
        pid (int): unique process ID
        post_file_prefix (str): unique prefix to name the post-processed file
        post_options (str): options for post-processing
        result (np.array): simulation result to be stored in database
        scheduler_type (str): type of job scheduler (pbs or slurm)
        simulation_input_template (str): path to simulation input template (e.g. dat-file)
        singularity (bool): flag for use of a singularity container
        workdir (path): path to working directory
    """

    def __init__(
        self,
        batch,
        driver_name,
        experiment_dir,
        experiment_name,
        job_id,
        num_procs,
        output_directory,
        singularity,
        database,
        cae_output_streaming,
        cluster_config,
        cluster_options,
        error_file,
        executable,
        input_file,
        log_file,
        num_procs_post,
        output_file,
        output_prefix,
        post_file_prefix,
        post_options,
        post_processor,
        cluster_job,
        scheduler_type,
        simulation_input_template,
        workdir,
        data_processor,
        gradient_data_processor,
        mpi_cmd,
    ):
        """Initialize MpiDriver object.

        Args:
            batch (int): current batch of driver calls.
            driver_name (str): name of the driver used for the analysis
            experiment_dir (path): path to QUEENS experiment directory
            experiment_name (str): name of QUEENS experiment
            job_id (int):  job ID as provided in database within range [1, n_jobs]
            num_procs (int): number of processors for processing
            output_directory (path): path to output directory (on remote computing resource)
            singularity (bool): flag for use of a singularity container
            database (obj): database object
            cae_output_streaming (bool): flag for additional streaming to given stream
            cluster_config (ClusterConfig): configuration data of cluster
            cluster_options (dict): cluster options for pbs or slurm
            error_file (path): path to error file
            executable (path): path to main executable of respective software (e.g. baci)
            input_file (path): path to input file
            log_file (path): path to log file
            num_procs_post (int): number of processors for post-processing
            output_file (path): path to output file
            output_prefix (str): output prefix
            post_file_prefix (str): unique prefix to name the post-processed files
            post_options (str): options for post-processing
            post_processor (path): path to post_processor
            cluster_job (bool): true if job is execute on cluster
            scheduler_type (str): type of job scheduler (pbs or slurm)
            simulation_input_template (path): path to simulation input template (e.g. dat-file)
            workdir (str): path to working directory
            data_processor (obj): instance of data processor class
            gradient_data_processor (obj): instance of data processor class for gradient data
            mpi_cmd (str): mpi command
        """
        super().__init__(
            batch=batch,
            driver_name=driver_name,
            experiment_dir=experiment_dir,
            experiment_name=experiment_name,
            job=None,
            job_id=job_id,
            num_procs=num_procs,
            output_directory=output_directory,
            result=None,
            gradient=None,
            database=database,
            post_processor=post_processor,
            gradient_data_processor=gradient_data_processor,
            data_processor=data_processor,
        )
        self.cae_output_streaming = cae_output_streaming
        self.cluster_job = cluster_job
        self.cluster_config = cluster_config
        self.cluster_options = cluster_options
        self.error_file = error_file
        self.executable = executable
        self.input_file = input_file
        self.log_file = log_file
        self.mpi_cmd = mpi_cmd
        self.num_procs_post = num_procs_post
        self.output_file = output_file
        self.output_prefix = output_prefix
        self.pid = None
        self.post_file_prefix = post_file_prefix
        self.post_options = post_options
        self.scheduler_type = scheduler_type
        self.simulation_input_template = simulation_input_template
        self.singularity = singularity
        self.workdir = workdir

    @classmethod
    def from_config_create_driver(
        cls,
        config,
        job_id,
        batch,
        driver_name,
        workdir=None,
        cluster_config=None,
        cluster_options=None,
    ):
        """Create Driver to run executable from input configuration.

        Set up required directories and files.

        Args:
            config (dict): dictionary containing configuration from QUEENS input file
            job_id (int): job ID as provided in database within range [1, n_jobs]
            batch (int): job batch number (multiple batches possible)
            driver_name (str): name of driver instance that should be realized
            workdir (str): path to working directory on remote resource
            cluster_config (ClusterConfig): configuration data of cluster
            cluster_options (dict): cluster options for pbs or slurm

        Returns:
            MpiDriver (obj): instance of MpiDriver class
        """
        experiment_name = config['global_settings'].get('experiment_name')

        database = DB_module.database  # pylint: disable=no-member

        # If multiple resources are passed an error is raised in the resources module.
        resource_name = list(config['resources'])[0]
        scheduler_name = config['resources'][resource_name]['scheduler']
        scheduler_options = config[scheduler_name]
        experiment_dir = pathlib.Path(scheduler_options['experiment_dir'])
        num_procs = scheduler_options.get('num_procs', 1)
        num_procs_post = scheduler_options.get('num_procs_post', 1)
        singularity = scheduler_options.get('singularity', False)
        scheduler_type = scheduler_options['scheduler_type']
        cluster_job = scheduler_type in VALID_CLUSTER_SCHEDULER_TYPES

        driver_options = config[driver_name]
        simulation_input_template = pathlib.Path(driver_options['input_template'])
        executable = pathlib.Path(driver_options['path_to_executable'])
        mpi_cmd = driver_options.get('mpi_cmd', 'mpirun --bind-to none -np')
        post_processor_str = driver_options.get('path_to_postprocessor', None)
        if post_processor_str:
            post_processor = pathlib.Path(post_processor_str)
        else:
            post_processor = None
        post_file_prefix = driver_options.get('post_file_prefix', None)
        post_options = driver_options.get('post_process_options', '')

        data_processor_name = driver_options.get('data_processor', None)
        if data_processor_name:
            data_processor = from_config_create_data_processor(config, data_processor_name)
            cae_output_streaming = False
        else:
            data_processor = None
            cae_output_streaming = True

        gradient_data_processor_name = driver_options.get('gradient_data_processor', None)
        if gradient_data_processor_name:
            gradient_data_processor = from_config_create_data_processor(
                config, gradient_data_processor_name
            )
        else:
            gradient_data_processor = None

        dest_dir = experiment_dir.joinpath(str(job_id))
        output_directory = dest_dir.joinpath('output')
        if not output_directory.is_dir():
            output_directory.mkdir(parents=True)

        output_prefix = experiment_name + '_' + str(job_id)
        output_file = output_directory.joinpath(output_prefix)

        file_extension_obj = pathlib.PurePosixPath(simulation_input_template)
        input_file_str = output_prefix + file_extension_obj.suffix
        input_file = dest_dir.joinpath(input_file_str)

        log_file = output_directory.joinpath(output_prefix + '.log')
        error_file = output_directory.joinpath(output_prefix + '.err')

        return cls(
            batch=batch,
            driver_name=driver_name,
            experiment_dir=experiment_dir,
            experiment_name=experiment_name,
            job_id=job_id,
            num_procs=num_procs,
            output_directory=output_directory,
            singularity=singularity,
            database=database,
            cae_output_streaming=cae_output_streaming,
            cluster_config=cluster_config,
            cluster_options=cluster_options,
            error_file=error_file,
            executable=executable,
            input_file=input_file,
            log_file=log_file,
            num_procs_post=num_procs_post,
            output_file=output_file,
            output_prefix=output_prefix,
            post_file_prefix=post_file_prefix,
            post_options=post_options,
            post_processor=post_processor,
            scheduler_type=scheduler_type,
            cluster_job=cluster_job,
            simulation_input_template=simulation_input_template,
            workdir=workdir,
            data_processor=data_processor,
            gradient_data_processor=gradient_data_processor,
            mpi_cmd=mpi_cmd,
        )

    def prepare_input_files(self):
        """Prepare input file on remote machine."""
        inject(self.job['params'], str(self.simulation_input_template), str(self.input_file))

    def run_job(self):
        """Run executable."""
        if self.cluster_job:
            returncode = self._run_job_cluster()
        else:
            returncode = self._run_job_local()

        # detect failed jobs
        if returncode:
            self.result = None
            self.job['status'] = 'failed'
        else:
            # save potential path set above and number of processes to database
            self.job['num_procs'] = self.num_procs
            self._save_job_in_db()

    def post_processor_job(self):
        """Post-process job."""
        output_file = '--file=' + str(self.output_file)
        target_file = '--output=' + str(self.output_directory.joinpath(self.post_file_prefix))
        post_processor_cmd = self._assemble_post_processor_cmd(output_file, target_file)

        _logger.debug("Start post-processor with command:")
        _logger.debug(post_processor_cmd)
        run_subprocess(
            post_processor_cmd,
            additional_error_message="Post-processing failed!",
            raise_error_on_subprocess_failure=False,
        )

    def _run_job_local(self):
        """Run executable locally via subprocess."""
        execute_cmd = self._assemble_execute_cmd_local()

        _logger.debug("Start executable with command:")
        _logger.debug(execute_cmd)
        returncode, self.pid, _, _ = run_subprocess(
            execute_cmd,
            subprocess_type='simulation',
            terminate_expr='PROC.*ERROR',
            loggername=__name__ + f'_{self.job_id}',
            log_file=str(self.log_file),
            error_file=str(self.error_file),
            streaming=self.cae_output_streaming,
            raise_error_on_subprocess_failure=False,
        )

        return returncode

    def _run_job_cluster(self):
        """Run executable on cluster."""
        if self.singularity:
            execute_cmd = self._assemble_execute_cmd_cluster_singularity()
        else:
            execute_cmd = self._assemble_execute_cmd_cluster_native()

        returncode, self.pid, stdout, stderr = run_subprocess(
            execute_cmd, subprocess_type='simple', raise_error_on_subprocess_failure=False
        )

        if not self.singularity:
            # override the pid with cluster scheduler id
            # if singularity: pid is handled by ClusterScheduler._submit_singularity
            self.pid = get_cluster_job_id(self.scheduler_type, stdout, VALID_PBS_SCHEDULER_TYPES)

        # redirect stdout/stderr output to log and error file
        with open(str(self.log_file), "a", encoding='utf-8') as text_file:
            print(stdout, file=text_file)
        with open(str(self.error_file), "a", encoding='utf-8') as text_file:
            print(stderr, file=text_file)

        return returncode

    def _assemble_execute_cmd_local(self):
        """Assemble execute command for local job (native or singularity).

        Returns:
            execute command
        """
        command_list = [
            self.mpi_cmd,
            str(self.num_procs),
            str(self.executable),
            str(self.input_file),
            str(self.output_file),
        ]

        return ' '.join(command_list)

    def _assemble_execute_cmd_cluster_singularity(self):
        """Assemble execute command within singularity.

        Returns:
            execute command within the singularity container
        """
        command_list = [
            'cd',
            str(self.workdir),
            r'&&',
            str(self.executable),
            str(self.input_file),
            str(self.output_prefix),
        ]

        return ' '.join(command_list)

    def _assemble_execute_cmd_cluster_native(self):
        """Assemble execute command for native cluster run.

        Returns:
            Slurm- or PBS-based jobscript submission command
        """
        self.cluster_options['job_name'] = f"{self.experiment_name}_{self.job_id}"
        self.cluster_options['DESTDIR'] = str(self.output_directory)
        self.cluster_options['EXE'] = str(self.executable)
        self.cluster_options['INPUT'] = str(self.input_file)
        self.cluster_options['OUTPUTPREFIX'] = self.output_prefix

        submission_script_path = str(self.experiment_dir.joinpath('jobfile.sh'))
        inject(self.cluster_options, self.cluster_config.jobscript_template, submission_script_path)

        command_list = [self.cluster_config.start_cmd, submission_script_path]

        return ' '.join(command_list)

    def _assemble_post_processor_cmd(self, output_file, target_file):
        """Assemble command for post-processing.

        Args:
            output_file (str): path with name to the simulation output files without the
                               file extension
            target_file (str): path with name of the post-processed file without the file extension

        Returns:
            post-processing command
        """
        if self.cluster_job and self.singularity:
            # no mpi necessary within the singularity container
            mpi_wrapper = ''
        else:
            mpi_wrapper = self.mpi_cmd + ' ' + str(self.num_procs_post)

        command_list = [
            mpi_wrapper,
            str(self.post_processor),
            output_file,
            self.post_options,
            target_file,
        ]

        return ' '.join(command_list)

    def __str__(self):
        """String of mpi driver.

        Returns:
            str: String version of the mpi driver
        """
        name = "MpiDriver."

        print_dict = {
            "Executable": self.executable,
            "MPI command": self.mpi_cmd,
            "Process ID": self.pid,
            "Scheduler type": self.scheduler_type,
            "Singularity": self.singularity,
            "Cluster job": self.cluster_job,
            "Post-processor": self.post_processor,
            "Number of procs (main)": self.num_procs,
            "Number of procs (post)": self.num_procs_post,
        }
        return get_str_table(name, print_dict)
