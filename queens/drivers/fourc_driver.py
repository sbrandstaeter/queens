"""Driver to run fourc."""

import logging

from queens.drivers.jobscript_driver import JobscriptDriver
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)

_MPI_COMMAND = """
{{ mpi_cmd }} -np {{ num_procs }} {{ executable }} {{ input_file }} {{ output_file }}
if [ ! -z "{{ post_processor }}" ]
then
  {{ mpi_cmd }} -np {{ num_procs }} {{ post_processor }} --file={{ output_file }} {{ post_options }}
fi
"""


class FourcDriver(JobscriptDriver):
    """Driver to run fourc."""

    @log_init_args
    def __init__(
        self,
        input_template,
        executable,
        files_to_copy=None,
        data_processor=None,
        gradient_data_processor=None,
        post_processor=None,
        post_process_options="",
        mpi_cmd="/usr/bin/mpirun --bind-to none",
    ):
        """Initialize FourcDriver object.

        Args:
            input_template (str, Path): path to simulation input template
            executable (str, Path): path to main executable of respective software
            files_to_copy (list, opt): files or directories to copy to experiment_dir
            data_processor (obj, opt): instance of data processor class
            gradient_data_processor (obj, opt): instance of data processor class for gradient data
            post_processor (path, opt): path to post_processor
            post_process_options (str, opt): options for post-processing
            mpi_cmd (str, opt): mpi command
        """
        extra_options = {
            "post_processor": post_processor,
            "post_process_options": post_process_options,
            "mpi_cmd": mpi_cmd,
        }
        super().__init__(
            input_template=input_template,
            jobscript_template=_MPI_COMMAND,
            executable=executable,
            files_to_copy=files_to_copy,
            data_processor=data_processor,
            gradient_data_processor=gradient_data_processor,
            extra_options=extra_options,
        )
