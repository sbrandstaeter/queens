"""Wrapped functions of subprocess stdlib module."""
import logging
import subprocess

_logger = logging.getLogger(__name__)

from pqueens.utils.exceptions import SubprocessError
from pqueens.utils.logger_settings import finish_job_logger, get_job_logger, job_logging
from pqueens.utils.valid_options_utils import get_option

# Currently allowed errors that might appear but have no effect on subprocesses
_allowed_errors = ["Invalid MIT-MAGIC-COOKIE-1 key", "No protocol specified"]


def run_subprocess(command_string, **kwargs):
    """Run a system command outside of the Python script.

    Different implementations dependent on subprocess_type.

    Args:
        command_string (str): Command string that should be run outside of Python
        subprocess_type (str): subprocess_type of run_subprocess from utils
        loggername (str): loggername for logging module
        terminate_expr (str): regex to search in stdout on which subprocess will terminate
        output_file (str): output directory + filename-stem to write logfiles
        error_file (str): output directory + filename-stem to write error files
        stream (str): streaming output to given stream
    Returns:
        process_returncode (int): code for execution success of subprocess
        process_id (int): process id that was assigned to process
        stdout (str): standard output content
        stderr (str): standard error content
    """
    # default subprocess type is "simple"
    subprocess_type = kwargs.get('subprocess_type', 'simple')

    subprocess_specific = _get_subprocess(subprocess_type)
    return subprocess_specific(command_string, **kwargs)


def _get_subprocess(desired_subprocess):
    """Choose subprocess implementation by subprocess_type.

    Args:
        desired_subprocess (str): subprocess type of run_subprocess
    Returns:
        function object (obj): function object for implementation type of run_subprocess from utils
    """
    valid_subprocess_types = {
        'simple': _run_subprocess_simple,
        'simulation': _run_subprocess_simulation,
        'submit': _run_subprocess_submit_job,
        'remote': _run_subprocess_remote,
    }
    return get_option(
        valid_subprocess_types, desired_subprocess, error_message="Invalid subprocess type!"
    )


def _run_subprocess_simple(command_string, **kwargs):
    """Run a system command outside of the Python script.

    return stderr and stdout
    Args:
        command_string (str): command, that will be run in subprocess
    Returns:
        process_returncode (int): code for success of subprocess
        process_id (int): unique process id, the subprocess was assigned on computing machine
        stdout (str): standard output content
        stderr (str): standard error content
    """
    process = subprocess.Popen(
        command_string,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    process_id = process.pid
    process_returncode = process.returncode

    _raise_or_warn_error(command_string, stdout, stderr, **kwargs)
    return process_returncode, process_id, stdout, stderr


def _run_subprocess_simulation(command_string, **kwargs):
    """Run a system command outside of the Python script.

    Log errors and stdout-return to initialized logger during runtime. Terminate subprocess if
    regular expression pattern is found in stdout.

    Args:
        command_string (str): command, that will be run in subprocess
        terminate_expr (str): regular expression to terminate subprocess
        logger (str): logger name to write to. Should be configured previously
        log_file (str): path to log file
        error_file (str): path to error file
    Returns:
        process_returncode (int): code for success of subprocess
        process_id (int): unique process id, the subprocess was assigned on computing machine
        stdout (str): always None
        stderr (str): standard error content
    """
    # get input data
    logger_name = kwargs.get('loggername')
    log_file = kwargs.get('log_file')
    error_file = kwargs.get('error_file')
    streaming = kwargs.get('streaming')
    terminate_expr = kwargs.get('terminate_expr')

    # setup job logging and get job logger as well as handlers
    joblogger, lfh, efh, sh = get_job_logger(logger_name, log_file, error_file, streaming)

    # run subprocess
    process = subprocess.Popen(
        command_string,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True,
    )

    # actual logging of job
    stderr = job_logging(command_string, process, joblogger, terminate_expr)

    # stdout should be empty. nevertheless None is returned by default to keep the interface to
    # run_subprocess consistent.
    stdout = None

    # get ID and returncode of subprocess
    process_id = process.pid
    process_returncode = process.returncode

    # close and remove file handlers (to prevent OSError: [Errno 24] Too many open files)
    finish_job_logger(joblogger, lfh, efh, sh)

    _raise_or_warn_error(command_string, stdout, stderr, **kwargs)
    return process_returncode, process_id, stdout, stderr


def _run_subprocess_submit_job(command_string, **kwargs):
    """Submit a system command (drop errors and stdout-return).

    Args:
        command_string (str): command, that will be run in subprocess
    Returns:
        process_returncode (int): always None here. this function does not wait for
                                    subprocess to finish.
        process_id (int): unique process id, the subprocess was assigned on computing machine
        stdout (str): always None
        stderr (str): always None
    """
    process = subprocess.Popen(
        command_string,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True,
    )

    process_id = process.pid
    # to keep the interface for run_subprocess consistent first return value is None (as it would
    # be, when you just submit the subprocess and not wait)
    process_returncode = None

    # stdout and stderr cannot be written in this state. nevertheless None is returned by default.
    stdout = None
    stderr = None

    return process_returncode, process_id, stdout, stderr


def _run_subprocess_remote(command_string, **kwargs):
    """Run a system command on a remote machine via ssh.

    Args:
        command_string (str): command, that will be run in subprocess
    Returns:
        process_returncode (int): code for success of subprocess
        process_id (int): unique process id, the subprocess was assigned on computing machine
        stdout (str): standard output content
        stderr (str): standard error content
    """
    remote_user = kwargs.get("remote_user", None)
    if not remote_user:
        raise SubprocessError("Remote commands need remote username.")

    remote_address = kwargs.get("remote_address", None)
    if not remote_user:
        raise SubprocessError("Remote commands needs remote machine address.")

    command_string = f'ssh {remote_user}@{remote_address} "{command_string}"'
    process = subprocess.Popen(
        command_string,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    process_id = process.pid
    process_returncode = process.returncode
    _raise_or_warn_error(command_string, stdout, stderr, **kwargs)
    return process_returncode, process_id, stdout, stderr


def _raise_or_warn_error(command, stdout, stderr, **kwargs):
    """Raise or warn eventual exception if subprocess fails.

    Args:
        command (str): Command string
        stdout (str): Command output
        stderr (str): Error of the output
        raise_error_on_subprocess_failure (bool,optional): Raise or warn error defaults to True
        additional_error_message (str,optional): Additional error message to be displayed
        allowed_errors (lst,optional): List of strings to be removed from the error message
    """
    # Check for allowed error messages and remove them
    allowed_errors = kwargs.get("allowed_errors", [])

    stderr = _remove_allowed_errors(stderr, allowed_errors)
    if stderr:
        raise_error_on_subprocess_failure = kwargs.get('raise_error_on_subprocess_failure', True)
        additional_message = kwargs.get('additional_error_message', None)
        subprocess_error = SubprocessError.construct_error_from_command(
            command, stdout, stderr, additional_message
        )
        if raise_error_on_subprocess_failure:
            raise subprocess_error
        else:
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
    allowed_errors.extend(_allowed_errors)
    # Remove the allowed error messages from stderr
    for em in allowed_errors:
        stderr = stderr.replace(em, "")

    # Remove trailing spaces, tabs and newlines and check if an error message remains
    if "".join(stderr.split()) == "":
        stderr = ""

    return stderr
