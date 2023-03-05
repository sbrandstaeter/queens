"""Driver to run an executable with mpi."""

import logging
import os
import pathlib

from pqueens.data_processor import from_config_create_data_processor
from pqueens.drivers.dask_driver import Driver
from pqueens.utils.injector import inject_in_template, read_template
from pqueens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)


class MpiDriver(Driver):
    """Driver to run an executable with mpi.

    Attributes:
    """

    def __init__(
        self,
        executable,
        cae_output_streaming,
        post_file_prefix,
        post_options,
        post_processor,
        simulation_input_suffix,
        simulation_input_template,
        data_processor,
        gradient_data_processor,
        mpi_cmd,
        environment,
    ):
        """Initialize MpiDriver object.

        Args:
            cae_output_streaming (bool): flag for additional streaming to given stream
            executable (path): path to main executable of respective software (e.g. baci)
            post_file_prefix (str): unique prefix to name the post-processed files
            post_options (str): options for post-processing
            post_processor (path): path to post_processor
            simulation_input_template (path): path to simulation input template (e.g. dat-file)
            data_processor (obj): instance of data processor class
            gradient_data_processor (obj): instance of data processor class for gradient data
            mpi_cmd (str): mpi command
            environment (dict)
        """
        super().__init__(
            data_processor,
            gradient_data_processor,
            simulation_input_suffix,
            simulation_input_template,
        )
        self.executable = executable
        self.cae_output_streaming = cae_output_streaming
        self.mpi_cmd = mpi_cmd
        self.environment = environment
        self.pid = None
        self.post_file_prefix = post_file_prefix
        self.post_options = post_options
        self.post_processor = post_processor

    @classmethod
    def from_config_create_driver(
        cls,
        config,
        driver_name,
    ):
        """Create Driver to run executable from input configuration.

        Set up required directories and files.

        Args:
            config (dict): Dictionary containing configuration from QUEENS input file
            driver_name (str): Name of driver instance that should be realized

        Returns:
            MpiDriver (obj): Instance of MpiDriver class
        """
        driver_options = config[driver_name]
        simulation_input_suffix = pathlib.PurePosixPath(driver_options['input_template']).suffix
        simulation_input_template = read_template(driver_options['input_template'])
        executable = pathlib.Path(driver_options['path_to_executable'])
        mpi_cmd = driver_options.get('mpi_cmd', 'mpirun --bind-to none -np')
        post_processor_str = driver_options.get('path_to_postprocessor', None)
        if post_processor_str:
            post_processor = pathlib.Path(post_processor_str)
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

        # Remove conda from environment
        environment = os.environ.copy()
        path_variables = environment['PATH'].split(":")
        new_path_variables = []
        for path_variable in path_variables:
            if not "conda" in path_variable:
                new_path_variables.append(path_variable)
        environment['PATH'] = ':'.join(new_path_variables)

        return cls(
            executable=executable,
            cae_output_streaming=cae_output_streaming,
            post_file_prefix=post_file_prefix,
            post_options=post_options,
            post_processor=post_processor,
            simulation_input_suffix=simulation_input_suffix,
            simulation_input_template=simulation_input_template,
            data_processor=data_processor,
            gradient_data_processor=gradient_data_processor,
            mpi_cmd=mpi_cmd,
            environment=environment,
        )

    def run(self, sample_dict, num_procs, num_procs_post, experiment_dir, experiment_name):
        job_id = sample_dict.pop('job_id')
        job_dir, output_dir, output_file, input_file, log_file, error_file = self._manage_paths(
            job_id, experiment_dir, experiment_name
        )

        inject_in_template(sample_dict, self.simulation_input_template, str(input_file))

        self._run_executable(job_id, num_procs, input_file, output_file, log_file, error_file)
        self._run_post_processing(num_procs_post, output_file, output_dir)

        return self._get_results(output_dir)

    def _run_executable(self, job_id, num_procs, input_file, output_file, log_file, error_file):
        """Run executable."""
        execute_cmd = self._assemble_execute_cmd(num_procs, input_file, output_file)

        _logger.debug("Start executable with command:")
        _logger.debug(execute_cmd)

        run_subprocess(
            execute_cmd,
            subprocess_type='simulation',
            terminate_expr='PROC.*ERROR',
            loggername=__name__ + f'_{job_id}',
            log_file=str(log_file),
            error_file=str(error_file),
            streaming=self.cae_output_streaming,
            raise_error_on_subprocess_failure=False,
            environment=self.environment,
        )

    def _run_post_processing(self, num_procs_post, output_file, output_dir):
        """Run post-processing."""
        if self.post_processor:
            output_file = '--file=' + str(output_file)
            target_file = '--output=' + str(output_dir.joinpath(self.post_file_prefix))
            post_processor_cmd = self._assemble_post_processor_cmd(
                num_procs_post, output_file, target_file
            )

            _logger.debug("Start post-processor with command:")
            _logger.debug(post_processor_cmd)

            run_subprocess(
                post_processor_cmd,
                additional_error_message="Post-processing failed!",
                raise_error_on_subprocess_failure=True,
                environment=self.environment,
            )

    def _assemble_execute_cmd(self, num_procs, input_file, output_file):
        """Assemble execute command.

        Returns:
            execute command
        """
        command_list = [
            self.mpi_cmd,
            str(num_procs),
            str(self.executable),
            str(input_file),
            str(output_file),
        ]

        return ' '.join(command_list)

    def _assemble_post_processor_cmd(self, num_procs_post, output_file, target_file):
        """Assemble command for post-processing.

        Args:
            num_procs_post
            output_file (str): path with name to the simulation output files without the
                               file extension
            target_file (str): path with name of the post-processed file without the file extension

        Returns:
            post-processing command
        """
        command_list = [
            self.mpi_cmd,
            str(num_procs_post),
            str(self.post_processor),
            output_file,
            self.post_options,
            target_file,
        ]

        return ' '.join(command_list)
