"""Driver to run an executable with mpi."""
import logging
import time
from pathlib import Path, PurePosixPath

from pqueens.data_processor import from_config_create_data_processor
from pqueens.utils.injector import inject
from pqueens.utils.print_utils import get_str_table
from pqueens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)


class DaskDriver:
    """Driver to run an executable with mpi.

    Attributes:
        cae_output_streaming (bool): Flag for additional streaming to given
                                     stream.
        cluster_options (dict): Cluster options for pbs or slurm.
        error_file (path): Path to error file.
        executable (path): Path to main executable of respective software
                           (e.g. BACI).
        input_file (path): Path to input file.
        log_file (path): Path to log file.
        mpi_cmd (str): mpi command.
        num_procs_post (int): Number of processors for post-processing.
        output_file (path): Path to output file.
        output_prefix (str): Output prefix.
        pid (int): Unique process ID.
        post_file_prefix (str): Unique prefix to name the post-processed
                                file.
        post_options (str): Options for post-processing.
        result (np.array): simulation result
        simulation_input_template (str): Path to simulation input template
                                         (e.g. dat-file)
        experiment_dir (path): path to working directory
    """

    def __init__(
        self,
        batch,
        driver_name,
        experiment_dir,
        initial_working_dir,
        experiment_name,
        job_id,
        num_procs,
        cae_output_streaming,
        cluster_config,
        cluster_options,
        executable,
        num_procs_post,
        post_file_prefix,
        post_options,
        post_processor,
        simulation_input_template,
        data_processor,
        gradient_data_processor,
        mpi_cmd,
        job,
    ):
        """Initialize MpiDriver object.

        Args:
            batch (int): current batch of driver calls.
            driver_name (str): name of the driver used for the analysis
            experiment_dir (path): path to QUEENS experiment directory
            experiment_name (str): name of QUEENS experiment
            job_id (int):  job ID within range [1, n_jobs]
            num_procs (int): number of processors for processing
            cae_output_streaming (bool): flag for additional streaming to given stream
            cluster_config (ClusterConfig): configuration data of cluster
            cluster_options (dict): cluster options for pbs or slurm
            executable (path): path to main executable of respective software (e.g. baci)
            num_procs_post (int): number of processors for post-processing
            post_file_prefix (str): unique prefix to name the post-processed files
            post_options (str): options for post-processing
            post_processor (path): path to post_processor
            simulation_input_template (path): path to simulation input template (e.g. dat-file)
            data_processor (obj): instance of data processor class
            gradient_data_processor (obj): instance of data processor class for gradient data
            mpi_cmd (str): mpi command
        """
        self.batch = batch
        self.driver_name = driver_name
        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_name
        self.job = job
        self.job_id = job_id
        self.num_procs = num_procs
        self.result = None
        self.gradient = None
        self.post_processor = post_processor
        self.gradient_data_processor = gradient_data_processor
        self.data_processor = data_processor
        self.cae_output_streaming = cae_output_streaming
        self.cluster_config = cluster_config
        self.cluster_options = cluster_options
        self.executable = executable
        self.mpi_cmd = mpi_cmd
        self.num_procs_post = num_procs_post
        self.pid = None
        self.post_file_prefix = post_file_prefix
        self.post_options = post_options
        self.simulation_input_template = simulation_input_template
        self.initial_working_dir = initial_working_dir

        # these have to be set by update_directories once the other states are updated correctly
        self.output_directory = None
        self.working_dir = None
        self.output_prefix = None
        self.output_file = None
        self.input_file = None
        self.log_file = None
        self.error_file = None

    @classmethod
    def from_config_create_driver(
        cls,
        config,
        job_id,
        batch,
        driver_name,
        experiment_dir,
        initial_working_dir,
        job,
        cluster_config=None,
        cluster_options=None,
    ):
        """Create Driver to run executable from input configuration.

        Set up required directories and files.

        Args:
            config (dict): Dictionary containing configuration from QUEENS input file
            job_id (int): Job ID within range [1, n_jobs]
            batch (int): Job batch number (multiple batches possible)
            driver_name (str): Name of driver instance that should be realized
            initial_working_dir (str): Path to working directory on remote resource
            cluster_options (dict): Cluster options for pbs or slurm
            experiment_dir (path): path to QUEENS experiment directory
            cluster_config (ClusterConfig): configuration data of cluster

        Returns:
            MpiDriver (obj): Instance of MpiDriver class
        """
        experiment_name = config['global_settings'].get('experiment_name')

        driver_options = config[driver_name]
        num_procs = driver_options.get('num_procs', 1)
        num_procs_post = driver_options.get('num_procs_post', 1)
        simulation_input_template = Path(driver_options['input_template'])
        executable = Path(driver_options['path_to_executable'])
        mpi_cmd = driver_options.get('mpi_cmd', 'mpirun --bind-to none -np')
        post_processor_str = driver_options.get('path_to_postprocessor', None)
        if post_processor_str:
            post_processor = Path(post_processor_str)
        else:
            post_processor = None
        post_file_prefix = driver_options.get('post_file_prefix', None)
        post_options = driver_options.get('post_process_options', '')

        data_processor_name = driver_options.get('data_processor_name', None)
        if data_processor_name:
            data_processor = from_config_create_data_processor(config, data_processor_name)
            cae_output_streaming = False
        else:
            data_processor = None
            cae_output_streaming = True

        gradient_data_processor_name = driver_options.get('gradient_data_processor_name', None)
        if gradient_data_processor_name:
            gradient_data_processor = from_config_create_data_processor(
                config, gradient_data_processor_name
            )
        else:
            gradient_data_processor = None

        return cls(
            batch=batch,
            driver_name=driver_name,
            experiment_dir=experiment_dir,
            initial_working_dir=initial_working_dir,
            experiment_name=experiment_name,
            job_id=job_id,
            num_procs=num_procs,
            cae_output_streaming=cae_output_streaming,
            cluster_config=cluster_config,
            cluster_options=cluster_options,
            executable=executable,
            num_procs_post=num_procs_post,
            post_file_prefix=post_file_prefix,
            post_options=post_options,
            post_processor=post_processor,
            simulation_input_template=simulation_input_template,
            data_processor=data_processor,
            gradient_data_processor=gradient_data_processor,
            mpi_cmd=mpi_cmd,
            job=job,
        )

    def set_job(self, job_id, batch, job):
        """Set state of job.

        Args:
            job_id (int): Job ID within range [1, n_jobs]
            batch (int): Job batch number (multiple batches possible)
            job (dict): Container for all job related data
        """
        self.job_id = job_id
        self.batch = batch
        self.job = job

        (
            self.output_directory,
            self.working_dir,
            self.output_prefix,
            self.output_file,
            self.input_file,
            self.log_file,
            self.error_file,
        ) = self.update_directories()

    def update_directories(self):
        """Update directories depending on job_id."""
        job_dir = self.experiment_dir / str(self.job_id)
        output_directory = job_dir / 'output'
        output_directory.mkdir(parents=True, exist_ok=True)

        if self.initial_working_dir is None:
            working_dir = output_directory
        else:
            working_dir = self.initial_working_dir

        output_prefix = self.experiment_name + '_' + str(self.job_id)
        output_file = output_directory.joinpath(output_prefix)

        file_extension_obj = PurePosixPath(self.simulation_input_template)
        input_file_str = output_prefix + file_extension_obj.suffix
        input_file = job_dir.joinpath(input_file_str)

        log_file = output_directory.joinpath(output_prefix + '.log')
        error_file = output_directory.joinpath(output_prefix + '.err')

        return (
            output_directory,
            working_dir,
            output_prefix,
            output_file,
            input_file,
            log_file,
            error_file,
        )

    # ------ Core methods ----------------------------------------------------- #
    def pre_job_run_and_run_job(self):
        """Prepare and execute job run."""
        self.pre_job_run()
        self.run_job()

    def pre_job_run(self):
        """Prepare job run."""
        self.initialize_job()
        self.prepare_input_files()

    def post_job_run(self):
        """Post-process, data processing and finalize job."""
        if self.post_processor:
            self.post_processor_job()

        if self.gradient_data_processor:
            self.gradient_data_processor_job()

        if self.data_processor:
            self.data_processor_job()
        else:
            # set result to "no", if there
            # has not been any data-processing before
            self.result = 'no processed data result'

        self.finalize_job()

    # ------ Base class methods ------------------------------------------------ #
    def initialize_job(self):
        """Initialize job."""
        start_time = time.time()
        self.job['start_time'] = start_time

    def data_processor_job(self):
        """Extract data of interest from post-processed file."""
        # only proceed if this job did not fail
        if self.job['status'] != "failed":
            self.result = self.data_processor.get_data_from_file(str(self.output_directory))
            _logger.debug("Got result: %s", self.result)

    def gradient_data_processor_job(self):
        """Extract gradient data from post-processed file."""
        # only proceed if this job did not fail
        if self.job['status'] != "failed":
            self.gradient = self.gradient_data_processor.get_data_from_file(
                str(self.output_directory)
            )
            _logger.debug("Got gradient: %s", self.gradient)

    def finalize_job(self):
        """Finalize job."""
        if self.result is None:
            self.job['result'] = None
            self.job['gradient'] = None
            self.job['status'] = 'failed'
            self.job['end_time'] = time.time()
        else:
            self.job['result'] = self.result
            self.job['gradient'] = self.gradient
            self.job['status'] = 'complete'
            if self.job['start_time']:
                self.job['end_time'] = time.time()
                computing_time = self.job['end_time'] - self.job['start_time']
                _logger.info(
                    "Successfully completed job %s (No. of proc.: %s, computing time: %s s).\n",
                    self.job_id,
                    self.num_procs,
                    computing_time,
                )
            _logger.info("Finalized job %s.", self.job_id)

    def prepare_input_files(self):
        """Prepare input file on remote machine."""
        inject(self.job['params'], str(self.simulation_input_template), str(self.input_file))

    def run_job(self):
        """Run executable."""
        execute_cmd = self._assemble_execute_cmd()

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
            raise_error_on_subprocess_failure=True,
        )

        # detect failed jobs
        if returncode:
            self.result = None
            self.job['status'] = 'failed'
        else:
            # save number of processes used for mpirun
            self.job['num_procs'] = self.num_procs

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
            raise_error_on_subprocess_failure=True,
        )

    def _assemble_execute_cmd(self):
        """Assemble execute command.

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

    def _assemble_post_processor_cmd(self, output_file, target_file):
        """Assemble command for post-processing.

        Args:
            output_file (str): path with name to the simulation output files without the
                               file extension
            target_file (str): path with name of the post-processed file without the file extension

        Returns:
            post-processing command
        """
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
        """Represent mpi driver as a string.

        Returns:
            str: String version of the mpi driver
        """
        name = "MpiDriver."

        print_dict = {
            "Executable": self.executable,
            "MPI command": self.mpi_cmd,
            "Process ID": self.pid,
            "Post-processor": self.post_processor,
            "Number of procs (main)": self.num_procs,
            "Number of procs (post)": self.num_procs_post,
        }
        return get_str_table(name, print_dict)
