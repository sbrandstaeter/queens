"""Wrapped functions of subprocess stdlib module."""
import logging
import subprocess

from queens.utils.exceptions import SubprocessError
from queens.utils.logger_settings import finish_job_logger, get_job_logger, job_logging

_logger = logging.getLogger(__name__)

# Currently allowed errors that might appear but have no effect on subprocesses
_ALLOWED_ERRORS = ["Invalid MIT-MAGIC-COOKIE-1 key", "No protocol specified"]


def run_subprocess(
    command,
    raise_error_on_subprocess_failure=True,
    additional_error_message=None,
    allowed_errors=None,
):
    """Run a system command outside of the Python script.

    return stderr and stdout
    Args:
        command (str): command, that will be run in subprocess
        raise_error_on_subprocess_failure (bool, optional): Raise or warn error defaults to True
        additional_error_message (str, optional): Additional error message to be displayed
        allowed_errors (lst, optional): List of strings to be removed from the error message
    Returns:
        process_returncode (int): code for success of subprocess
        process_id (int): unique process id, the subprocess was assigned on computing machine
        stdout (str): standard output content
        stderr (str): standard error content
    """
    process = start_subprocess(command)

    stdout, stderr = process.communicate()
    process_id = process.pid
    process_returncode = process.returncode

    _raise_or_warn_error(
        command=command,
        stdout=stdout,
        stderr=stderr,
        raise_error_on_subprocess_failure=raise_error_on_subprocess_failure,
        additional_error_message=additional_error_message,
        allowed_errors=allowed_errors,
    )
    return process_returncode, process_id, stdout, stderr


def run_subprocess_remote(
    command,
    remote_connect,
    raise_error_on_subprocess_failure=True,
    additional_error_message=None,
    allowed_errors=None,
):
    """Run a system command on a remote machine.

    return stderr and stdout
    Args:
        command (str): command, that will be run on a remote machine.
        remote_connect (str): <user>@<hostname>
        raise_error_on_subprocess_failure (bool, optional): Raise or warn error defaults to True
        additional_error_message (str, optional): Additional error message to be displayed
        allowed_errors (lst, optional): List of strings to be removed from the error message
    Returns:
        process_returncode (int): code for success of subprocess
        process_id (int): unique process id, the subprocess was assigned on computing machine
        stdout (str): standard output content
        stderr (str): standard error content
    """
    command = f'ssh {remote_connect} "{command}"'
    return run_subprocess(
        command=command,
        raise_error_on_subprocess_failure=raise_error_on_subprocess_failure,
        additional_error_message=additional_error_message,
        allowed_errors=allowed_errors,
    )


def run_subprocess_with_logging(
    command,
    terminate_expression,
    logger_name,
    log_file,
    error_file,
    full_log_formatting=True,
    streaming=False,
    raise_error_on_subprocess_failure=True,
    additional_error_message=None,
    allowed_errors=None,
):
    """Run a system command outside of the Python script.

    Log errors and stdout-return to initialized logger during runtime. Terminate subprocess if
    regular expression pattern is found in stdout.

    Args:
        command (str): command, that will be run in subprocess
        terminate_expression (str): regular expression to terminate subprocess
        logger_name (str): logger name to write to. Should be configured previously
        log_file (str): path to log file
        error_file (str): path to error file
        full_log_formatting (bool): Flag to add logger metadata in the simulation logs
        streaming (bool, optional): Flag for additional streaming to stdout
        raise_error_on_subprocess_failure (bool, optional): Raise or warn error defaults to True
        additional_error_message (str, optional): Additional error message to be displayed
        allowed_errors (lst, optional): List of strings to be removed from the error message
    Returns:
        process_returncode (int): code for success of subprocess
        process_id (int): unique process id, the subprocess was assigned on computing machine
        stdout (str): always None
        stderr (str): standard error content
    """
    # setup job logging and get job logger as well as handlers
    job_logger, log_file_handle, error_file_handler, stream_handler = get_job_logger(
        logger_name=logger_name,
        log_file=log_file,
        error_file=error_file,
        streaming=streaming,
        full_log_formatting=full_log_formatting,
    )

    # run subprocess
    process = start_subprocess(command)

    # actual logging of job
    stderr = job_logging(
        command_string=command,
        process=process,
        job_logger=job_logger,
        terminate_expression=terminate_expression,
    )

    stdout = ""

    # get ID and returncode of subprocess
    process_id = process.pid
    process_returncode = process.returncode

    # close and remove file handlers (to prevent OSError: [Errno 24] Too many open files)
    finish_job_logger(
        job_logger=job_logger,
        lfh=log_file_handle,
        efh=error_file_handler,
        stream_handler=stream_handler,
    )

    _raise_or_warn_error(
        command=command,
        stdout=stdout,
        stderr=stderr,
        raise_error_on_subprocess_failure=raise_error_on_subprocess_failure,
        additional_error_message=additional_error_message,
        allowed_errors=allowed_errors,
    )
    return process_returncode, process_id, stdout, stderr


def start_subprocess(command):
    """Start subprocess.

    Args:
        command (str): command, that will be run in subprocess

    Returns:
         process (subprocess.Popen): subprocess object
    """
    process = subprocess.Popen(  # pylint: disable=consider-using-with
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True,
    )
    return process


def _raise_or_warn_error(
    command,
    stdout,
    stderr,
    raise_error_on_subprocess_failure,
    additional_error_message,
    allowed_errors,
):
    """Raise or warn eventual exception if subprocess fails.

    Args:
        command (str): Command string
        stdout (str): Command output
        stderr (str): Error of the output
        raise_error_on_subprocess_failure (bool): Raise or warn error defaults to True
        additional_error_message (str): Additional error message to be displayed
        allowed_errors (lst): List of strings to be removed from the error message
    """
    # Check for allowed error messages and remove them
    if allowed_errors is None:
        allowed_errors = []

    stderr = _remove_allowed_errors(stderr, allowed_errors)
    if stderr:
        subprocess_error = SubprocessError.construct_error_from_command(
            command, stdout, stderr, additional_error_message
        )
        if raise_error_on_subprocess_failure:
            raise subprocess_error
        _logger.warning(str(subprocess_error))


def _remove_allowed_errors(stderr, allowed_errors):
    """Remove allowed error messages from error output.

    Args:
        stderr (str): Error message
        allowed_errors (lst): Allowed error messages

    Returns:
        stderr (str): error message without allowed errors
    """
    # Add known exceptions
    allowed_errors.extend(_ALLOWED_ERRORS)
    # Remove the allowed error messages from stderr
    for error_message in allowed_errors:
        stderr = stderr.replace(error_message, "")

    # Remove trailing spaces, tabs and newlines and check if an error message remains
    if "".join(stderr.split()) == "":
        stderr = ""

    return stderr
