"""Convenience wrapper around Jobscript Driver."""

from queens.drivers.jobscript_driver import JobscriptDriver
from queens.utils.logger_settings import log_init_args

_JOBSCRIPT_TEMPLATE = (
    "{{ mpi_cmd }} -np {{ num_procs }} {{ executable }} {{ input_file }} {{ output_file }}"
)


class MpiDriver(JobscriptDriver):
    """Driver to run a generic MPI run."""

    @log_init_args
    def __init__(
        self,
        input_template,
        executable,
        parameters,
        files_to_copy=None,
        data_processor=None,
        gradient_data_processor=None,
        mpi_cmd="/usr/bin/mpirun --bind-to none",
    ):
        """Initialize FourcDriver object.

        Args:
            input_template (str, Path): path to simulation input template
            executable (str, Path): path to main executable of respective software
            parameters (Parameters): Parameters object
            files_to_copy (list, opt): files or directories to copy to experiment_dir
            data_processor (obj, opt): instance of data processor class
            gradient_data_processor (obj, opt): instance of data processor class for gradient data
            mpi_cmd (str, opt): mpi command
        """
        extra_options = {
            "mpi_cmd": mpi_cmd,
        }
        super().__init__(
            input_template=input_template,
            jobscript_template=_JOBSCRIPT_TEMPLATE,
            executable=executable,
            files_to_copy=files_to_copy,
            data_processor=data_processor,
            gradient_data_processor=gradient_data_processor,
            extra_options=extra_options,
            parameters=parameters,
        )
