"""
Wrapped functions of subprocess stdlib module.
"""
import subprocess
import logging
import time
import re


def run_subprocess(command_string, type='simple', loggername=None, expr=None):
    """
    Run a system command outside of the Python script anc log errors and
    stdout-return
    Args:
        command_string (str): Command string that should be run outside of Python
    Returns:
        type (str): type of run_subprocess from utils
        loggername (str): loggername for logging module
        expr (str): regex to search in sdtout on which subprocess will terminate

    """

    subprocess_specific = _get_subprocess(type)

    return subprocess_specific(command_string, logger=loggername, terminate_expr=expr)


def _get_subprocess(type):
    """
        Run a system command outside of the Python script and log errors and
        stdout-return
        Args:
            type (str): Type of run_subprocess
        Returns:
            function object (obj): type of run_subprocess from utils

    """

    if type == 'simple':
        return _run_subprocess_simple
    elif type == 'term_on_expr':
        return _run_subprocess_terminateexpression
    elif type == 'whitelist':
        return _run_subprocess_whitelist
    else:
        raise ValueError(f'subprocess type {type} not found.')


def _run_subprocess_simple(command_string, terminate_expr=None, logger=None):
    """
        Run a system command outside of the Python script drop errors and
        stdout-return
        Args:
            command_string (str): command, that will be run in subprocess
        Returns:
            process_returncode (int): code for success of subprocess
            process_id (int): unique process id, the subprocess was assigned on computing machine

    """

    process = subprocess.Popen(
        command_string,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True,
    )
    _, _ = process.communicate()
    process_id = process.pid
    process_returncode = process.returncode
    return process_returncode, process_id


def _run_subprocess_terminateexpression(command_string, terminate_expr=None, logger=None):
    """
        Run a system command outside of the Python script. log errors and stdout-return to
        initialized logger. Terminate subprocess if regular expression pattern
        is found in stdout.
        Args:
            command_string (str): command, that will be run in subprocess
            terminate_expr (str): regular expression to terminate subprocess
            logger (str): logger name to write to. Should be configured previously
        Returns:
            process_returncode (int): code for success of subprocess
            process_id (int): unique process id, the subprocess was assigned on computing machine

    """
    joblogger = logging.getLogger(logger)
    process = subprocess.Popen(
        command_string,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True,
    )
    joblogger.info('run_subprocess started with:\n')
    joblogger.info(command_string + '\n')
    for line in iter(process.stdout.readline, b''):  # b'\n'-separated lines
        if line == '' and process.poll() is not None:
            joblogger.info(
                'subprocess.Popen-info: stdout is finished and process.poll() ' 'not None.\n'
            )
            stdout, stderr = process.communicate()
            joblogger.info(stdout + '\n')
            if stderr:
                joblogger.error('error message (if provided) follows:\n')
                joblogger.error(stderr)
            break
        if terminate_expr:
            if re.search(terminate_expr, line):
                joblogger.warning('run_subprocess detected terminate expression:\n')
                joblogger.error(line)
                time.sleep(2)
                if process.poll() is None:
                    # log terminate command
                    joblogger.warning('running job will be terminated.\n')
                    process.terminate()
                    time.sleep(2)
                continue
        joblogger.info(line)

    # stdout, stderr = process.communicate()
    process_id = process.pid
    process_returncode = process.returncode
    return process_returncode, process_id


def _run_subprocess_whitelist(command_string, terminate_expr=None, logger=None):
    """
        Run a system command outside of the Python script. log errors and stdout-return to
        initialized logger. Ignore predefined whitelisted errors. Manipulate returncode in case
        of whitelisted error message.
        Args:
            command_string (str): command, that will be run in subprocess
            logger (str): logger name to write to. Should be configured previously
        Returns:
            process_returncode (int): code for success of subprocess
            process_id (int): unique process id, the subprocess was assigned on computing machine

    """
    joblogger = logging.getLogger(logger)
    process = subprocess.Popen(
        command_string,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True,
    )
    joblogger.info('run_subprocess started with:\n')
    joblogger.info(command_string + '\n')
    stdout, stderr = process.communicate()
    for line in stdout:
        joblogger.info(line)
    for line in stderr:
        joblogger.error(line)
    process_id = process.pid
    process_returncode = process.returncode
    # TODO: fix this hack
    # For the second call of remote_main.py with the --post=true flag
    # (see the jobscript_slurm_queens.sh), the workdir does not exist anymore.
    # Therefore, change directory in command_list ("cd self.workdir") does throw an error.
    # We catch this error to detect that we are in a postprocessing call of the driver.

    # TODO: fix directory handling on clusters
    # These checks will set process_returncode to success in case of previously defined error
    # messages
    if re.fullmatch(
        r'/bin/sh: line 0: cd: /scratch/SLURM_\d+: No such file or directory\n', stderr
    ):
        process_returncode = 0
    elif re.fullmatch(
        r'/bin/sh: line 0: cd: /scratch/PBS_\d+.master.cluster: No such file or directory\n',
        stderr,
    ):
        process_returncode = 0
    return process_returncode, process_id
