"""Driver to run an executable with mpi."""

import logging
from pathlib import Path

from pqueens.drivers.dask_driver import Driver
from pqueens.utils.injector import inject
from pqueens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)


class MpiDriver(Driver):
    """Driver to run an executable with mpi.

    Attributes:
        executable (path): path to main executable of respective software
        cae_output_streaming (bool): flag for additional streaming to given stream
        post_file_prefix (str): unique prefix to name the post-processed files
        post_options (str): options for post-processing
        post_processor (path): path to post_processor
        mpi_cmd (str): mpi command
    """

    def __init__(
        self,
        input_template,
        path_to_executable,
        post_file_prefix=None,
        post_process_options='',
        path_to_postprocessor=None,
        data_processor=None,
        gradient_data_processor=None,
        mpi_cmd='/usr/bin/mpirun --bind-to none -np',
    ):
        """Initialize MpiDriver object.

        Args:
            input_template (str, Path): path to simulation input template
            path_to_executable (str, Path): path to main executable of respective software
            post_file_prefix (str, opt): unique prefix to name the post-processed files
            post_process_options (str, opt): options for post-processing
            path_to_postprocessor (path, opt): path to post_processor
            data_processor (obj, opt): instance of data processor class
            gradient_data_processor (obj, opt): instance of data processor class for gradient data
            mpi_cmd (str, opt): mpi command
        """
        super().__init__(
            input_template,
            data_processor,
            gradient_data_processor,
        )
        self.executable = Path(path_to_executable)
        self.post_processor = Path(path_to_postprocessor) if path_to_postprocessor else None
        self.cae_output_streaming = not data_processor
        self.mpi_cmd = mpi_cmd
        self.post_file_prefix = post_file_prefix
        self.post_options = post_process_options

    def run(self, sample_dict, num_procs, num_procs_post, experiment_dir, experiment_name):
        """Run the driver.

        Args:
            sample_dict (dict): Dict containing sample and job id
            num_procs (int): number of cores
            num_procs_post (int): number of cores for post-processing
            experiment_name (str): name of QUEENS experiment.
            experiment_dir (Path): Path to QUEENS experiment directory.

        Returns:
            Result and potentially the gradient
        """
        job_id = sample_dict.pop('job_id')
        _, output_dir, output_file, input_file, log_file, error_file = self._manage_paths(
            job_id, experiment_dir, experiment_name
        )

        inject(sample_dict, experiment_dir / self.simulation_input_template.name, str(input_file))

        self._run_executable(job_id, num_procs, input_file, output_file, log_file, error_file)
        self._run_post_processing(num_procs_post, output_file, output_dir)

        return self._get_results(output_dir)

    def _run_executable(self, job_id, num_procs, input_file, output_file, log_file, error_file):
        """Run executable.

        Args:
            job_id (int): Job id
            num_procs (int): number of cores
            input_file (Path): Path to input file
            output_file (Path): Path to output file(s)
            log_file (Path): Path to log file
            error_file (Path): Path to error file
        """
        execute_cmd = self._assemble_execute_cmd(num_procs, input_file, output_file)

        run_subprocess(
            execute_cmd,
            subprocess_type='simulation',
            terminate_expr='PROC.*ERROR',
            loggername=__name__ + f'_{job_id}',
            log_file=str(log_file),
            error_file=str(error_file),
            streaming=self.cae_output_streaming,
        )

    def _run_post_processing(self, num_procs_post, output_file, output_dir):
        """Run post-processing.

        Args:
            num_procs_post (int): number of cores for post-processing
            output_file (Path): Path to output file(s)
            output_dir (Path): Path to output directory
        """
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
            )

    def _assemble_execute_cmd(self, num_procs, input_file, output_file):
        """Assemble execute command.

        Args:
            num_procs (int): number of cores
            input_file (Path): Path to input file
            output_file (Path): Path to output file(s)

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
            num_procs_post (int): number of cores for post-processing
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
