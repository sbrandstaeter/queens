"""
Wrapped functions of subprocess stdlib module.
"""
import subprocess
import logging
import time
import re
import io
import sys


def run_subprocess(command_string, **kwargs):
    """
    Run a system command outside of the Python script. different implementations dependent on
    subprocess_type

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
    """
    Choose subprocess implementation by subprocess_type
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
        raise ValueError(f'subprocess_type {subprocess_type} not found.')


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
            log_file (str): path to log file
            error_file (str): path to error file
        Returns:
            process_returncode (int): code for success of subprocess
            process_id (int): unique process id, the subprocess was assigned on computing machine
            stdout (str): always None
            stderr (str): standard error content

    """
    logger = kwargs.get('loggername')
    terminate_expr = kwargs.get('terminate_expr')
    log_file = kwargs.get('log_file')
    error_file = kwargs.get('error_file')
    streaming = kwargs.get('streaming')

    joblogger = logging.getLogger(logger)

    ff = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    joblogger.setLevel(logging.INFO)

    # job logger configuration. This python code is run in parallel for cluster runs with
    # singularity, so each processor logs his own file
    fh = logging.FileHandler(log_file, mode='w', delay=False)
    fh.setLevel(logging.INFO)
    fh.terminator = ''
    efh = logging.FileHandler(error_file, mode='w', delay=False)
    efh.setLevel(logging.ERROR)
    efh.terminator = ''
    fh.setFormatter(ff)
    efh.setFormatter(ff)
    joblogger.addHandler(fh)
    joblogger.addHandler(efh)

    # additional streaming to given stream, if required
    # import pdb; pdb.set_trace()
    if streaming:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(logging.INFO)
        sh.terminator = ''
        sh.setFormatter(fmt=None)
        joblogger.addHandler(sh)

    joblogger = logging.getLogger(logger)
    process = subprocess.Popen(
        command_string,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True,
    )

    # actual logging
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
                # will be found also if program terminates itself properly
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

    # stdout should be empty. nevertheless None is returned by default to keep the interface to
    # run_subprocess consistent.
    stdout = None

    # we need to close the FileHandlers to prevent OSError: [Errno 24] Too many open files
    fh.close()
    efh.close()
    joblogger.removeHandler(fh)
    joblogger.removeHandler(efh)
    if streaming:
        sh.close()
        joblogger.removeHandler(sh)

    return process_returncode, process_id, stdout, stderr


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
    """
        Run a system command outside of the Python script on a remote machine via ssh.
        return stderr and stdout
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
        raise ValueError("Remote commands need remote username.")

    remote_address = kwargs.get("remote_address", None)
    if not remote_user:
        raise ValueError("Remote commands needs remote machine address.")

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
    return process_returncode, process_id, stdout, stderr
