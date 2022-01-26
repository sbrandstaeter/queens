"""Wrapped functions of subprocess stdlib module."""
import logging
import subprocess

_logger = logging.getLogger(__name__)

from pqueens.utils.logger_settings import finish_job_logger, get_job_logger, job_logging

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


def _get_subprocess(subprocess_type):
    """Choose subprocess implementation by subprocess_type.

    Args:
        subprocess_type (str): subprocess_type of run_subprocess
    Returns:
        function object (obj): function object for implementation type of run_subprocess from utils
    """
    if subprocess_type == 'simple':
        return _run_subprocess_simple
    elif subprocess_type == 'simulation':
        return _run_subprocess_simulation
    elif subprocess_type == 'submit':
        return _run_subprocess_submit_job
    elif subprocess_type == 'remote':
        return _run_subprocess_remote
    else:
        raise SubprocessError(f'subprocess_type {subprocess_type} not found.')


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
        raise_error (bool,optional): Raise or warn error defaults to True
        additional_error_message (str,optional): Additional error message to be displayed
        allowed_errors (lst,optional): List of strings to be removed from the error message
    """
    # Check for allowed error messages and remove them
    allowed_errors = kwargs.get("allowed_errors", [])

    stderr = _remove_allowed_errors(stderr + "\n", allowed_errors)
    if stderr:
        raise_error = kwargs.get('raise_error', True)
        additional_message = kwargs.get('additional_error_message', None)
        subprocess_error = SubprocessError.construct_error_from_command(
            command, stdout, stderr, additional_message
        )
        if raise_error:
            raise subprocess_error
        else:
            _logger.warning(subprocess_error.message)


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


class SubprocessError(Exception):
    """Custom error class for the QUEENS subprocess wrapper."""

    def __init__(
        self,
        message,
        command=None,
        command_output=None,
        error_message=None,
        additional_message=None,
    ):
        """Initialize SubprocessError.

        Do not create an error with this method. Instead use construct_error from command method.
        This makes it easier to handle this error within a context.

        Args:
            message (str): Error message
            command (str): Command used that raised the error. Defaults to None.
            command_output (str): Command output. Defaults to None.
            error_message (str): Error message of the command. Defaults to None.
            additional_message (str, optional): Additional message to pass. Defaults to None.
        """
        self.command = command
        self.command_output = command_output
        self.error_message = error_message
        self.additional_message = additional_message
        self.message = message
        super().__init__(self.message)

    @classmethod
    def construct_error_from_command(
        cls, command, command_output, error_message, additional_message=""
    ):
        """Construct a Subprocess error from a command and its outputs.

        Args:
            command (str): Command used that raised the error
            command_output (str): Command output
            error_message (str): Error message of the command
            additional_message (str, optional): Additional message to pass

        Returns:
            SubprocessError
        """
        message = cls.construct_error_message(
            command, command_output, error_message, additional_message
        )
        return cls(
            message,
            command=command,
            command_output=command_output,
            error_message=error_message,
            additional_message=additional_message,
        )

    @staticmethod
    def construct_error_message(command, command_output, error_message, additional_message):
        """Construct the error message based on the command and its outputs.

        Args:
            command (str): Command used that raised the error
            command_output (str): Command output
            error_message (str): Error message of the command
            additional_message (str, optional): Additional message to pass

        Returns:
            message (str): Error message that is display once the error is raised
        """
        message = "\n\nQUEENS' subprocess wrapper caught the following error:\n"
        message += error_message
        message += "\n\n\nwith commandline output:\n"
        message += str(command_output)
        message += "\n\n\nwhile executing the command:\n" + command
        if additional_message:
            message += '\n\n' + additional_message
        return message
