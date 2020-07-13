"""
Wrapped functions of subprocess stdlib module.
"""
import subprocess
import logging
import time
import re
import io


def run_subprocess(command_string, **kwargs):
    """
    Run a system command outside of the Python script and log errors and
    stdout-return
    Args:
        command_string (str): Command string that should be run outside of Python
        type (str): type of run_subprocess from utils
        loggername (str): loggername for logging module
        terminate_expr (str): regex to search in sdtout on which subprocess will terminate
    Returns:
        process_returncode (int): code for execution success of subprocess
        process_id (int): process id that was assigned to process

    """

    type = kwargs.get('type')

    if not type:
        type = 'simple'

    subprocess_specific = _get_subprocess(type)

    return subprocess_specific(command_string, **kwargs)


def _get_subprocess(type):
    """
        Chose subprocess implementation by type
        Args:
            type (str): Type of run_subprocess
        Returns:
            function object (obj): function object for implementation type of run_subprocess from
                                    utils

    """

    if type == 'simple':
        return _run_subprocess_simple
    elif type == 'simulation':
        return _run_subprocess_simulation
    elif type == 'submit':
        return _run_subprocess_submit_job
    else:
        raise ValueError(f'subprocess type {type} not found.')


def _run_subprocess_simple(command_string, **kwargs):
    """
        Run a system command outside of the Python script. return stderr and stdout
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
    return process_returncode, process_id, stdout, stderr


def _run_subprocess_simulation(command_string, **kwargs):
    """
        Run a system command outside of the Python script. log errors and stdout-return to
        initialized logger during runtime. Terminate subprocess if regular expression pattern
        is found in stdout.
        Args:
            command_string (str): command, that will be run in subprocess
            terminate_expr (str): regular expression to terminate subprocess
            logger (str): logger name to write to. Should be configured previously
        Returns:
            process_returncode (int): code for success of subprocess
            process_id (int): unique process id, the subprocess was assigned on computing machine

    """
    logger = kwargs.get('loggername')
    terminate_expr = kwargs.get('terminate_expr')

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
                'subprocess.Popen() -info: stdout is finished and process.poll() not None.\n'
            )
            # This line waits for termination and puts together stdout not yet consumed from the
            # stream by the logger and finally the stderr.
            stdout, stderr = process.communicate()
            # following line should never really do anything. We want to log all that was
            # written to stdout even after program was terminated.
            joblogger.info(stdout + '\n')
            if stderr:
                joblogger.error('error message (if provided) follows:\n')
                for errline in io.StringIO(stderr):
                    joblogger.error(errline)
            break
        if terminate_expr:
            # two seconds in time.sleep(2) are arbitrary. Feel free to tune it to your needs.
            if re.search(terminate_expr, line):
                joblogger.warning('run_subprocess detected terminate expression:\n')
                joblogger.error(line)
                # give program the chance to terminate by itself, because terminate expression
                # will be found also if program terminates correctly
                time.sleep(2)
                if process.poll() is None:
                    # log terminate command
                    joblogger.warning('running job will be terminated by QUEENS.\n')
                    process.terminate()
                    # wait before communicate call which gathers all the output
                    time.sleep(2)
                continue
        joblogger.info(line)

    process_id = process.pid
    process_returncode = process.returncode

    return process_returncode, process_id


def _run_subprocess_submit_job(command_string, **kwargs):
    """
        submit a system command outside of the Python script drop errors and
        stdout-return
        Args:
            command_string (str): command, that will be run in subprocess
        Returns:
            process_returncode (int): always None here. this function does not wait for
                                        subprocess to finish.
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

    process_id = process.pid
    # to keep the interface for run_subprocess consistent first return value is None (as it would
    # be, when you just submit the subprocess and not wait)
    process_returncode = None
    return process_returncode, process_id
