"""Logging in QUEENS."""

import io
import logging
import re
import sys
import time


class LogFilter(logging.Filter):
    """Filters (lets through) all messages with level <= LEVEL."""

    def __init__(self, level):
        """Initiate the logging filter.

        Args:
            level (int): Logging level
        """
        super().__init__()
        self.level = level

    def filter(self, record):
        """Filter the logging record.

        Args:
            record (LogRecord obj): Logging record object

        Returns:
            (LogRecord obj): Filter logging record
        """
        return record.levelno <= self.level


def setup_basic_logging(output_dir, experiment_name):
    """Setup basic logging.

    Args:
        output_dir (Path): output directory where to save the log-file
        experiment_name (str): experiment name used as file name for the log-file
    """
    file_level_min = logging.DEBUG
    console_level_min = logging.INFO

    # setup format for logging to file
    logging_file_path = output_dir.joinpath(experiment_name + ".log")
    logging.basicConfig(
        level=file_level_min,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename=logging_file_path,
        filemode='w',
    )

    console_stdout = logging.StreamHandler(stream=sys.stdout)
    console_stderr = logging.StreamHandler(stream=sys.stderr)

    # messages lower than and including WARNING go to stdout
    log_filter = LogFilter(logging.WARNING)
    console_stdout.addFilter(log_filter)
    console_stdout.setLevel(console_level_min)

    # messages >= ERROR or messages >= CONSOLE_LEVEL_MIN if CONSOLE_LEVEL_MIN > ERROR go to stderr
    console_stderr.setLevel(max(console_level_min, logging.ERROR))

    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console_stdout.setFormatter(formatter)
    console_stderr.setFormatter(formatter)

    # add the handlers to the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(console_stdout)
    root_logger.addHandler(console_stderr)

    # deactivate logging for specific modules
    logging.getLogger('arviz').setLevel(logging.CRITICAL)
    logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
    logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
    logging.getLogger('numba').setLevel(logging.CRITICAL)


def setup_cluster_logging():
    """Setup cluster logging."""
    level_min = logging.INFO

    logging.basicConfig(
        level=level_min,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
    )

    console_stdout = logging.StreamHandler(stream=sys.stdout)
    console_stderr = logging.StreamHandler(stream=sys.stderr)

    # messages lower than and including WARNING go to stdout
    log_filter = LogFilter(logging.WARNING)
    console_stdout.addFilter(log_filter)
    console_stdout.setLevel(level_min)

    # messages >= ERROR or messages >= CONSOLE_LEVEL_MIN if CONSOLE_LEVEL_MIN > ERROR go to stderr
    console_stderr.setLevel(max(level_min, logging.ERROR))

    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console_stdout.setFormatter(formatter)
    console_stderr.setFormatter(formatter)

    # add the handlers to the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(console_stdout)
    root_logger.addHandler(console_stderr)


def get_job_logger(logger_name, log_file, error_file, streaming, propagate=False):
    """Setup job logging and get job logger.

    Args:
        logger_name (str): logger name
        log_file (path): path to log file
        error_file (path): path to error file
        streaming (bool): flag for additional streaming to given stream
        propagate (bool): flag for propagation of stream (default: false)
    """
    # get job logger
    joblogger = logging.getLogger(logger_name)

    # define formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # set level
    joblogger.setLevel(logging.INFO)

    # set option to propagate (default: false)
    joblogger.propagate = propagate

    # add handlers for log and error file (remark: python code is run in parallel
    # for cluster runs with singularity; thus, each processor logs his own file.)
    lfh = logging.FileHandler(log_file, mode='w', delay=False)
    lfh.setLevel(logging.INFO)
    lfh.terminator = ''
    lfh.setFormatter(formatter)
    joblogger.addHandler(lfh)
    efh = logging.FileHandler(error_file, mode='w', delay=False)
    efh.setLevel(logging.ERROR)
    efh.terminator = ''
    efh.setFormatter(formatter)
    joblogger.addHandler(efh)

    # add handler for additional streaming to given stream, if required
    if streaming:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.terminator = ''
        stream_handler.setFormatter(fmt=None)
        joblogger.addHandler(stream_handler)
    else:
        stream_handler = None

    # return job logger and handlers
    return joblogger, lfh, efh, stream_handler


def job_logging(command_string, process, joblogger, terminate_expr):
    """Actual logging of job.

    Args:
        command_string (str): Command string for the subprocess
        process (obj): subprocess object
        joblogger (obj): job logger object
        terminate_expr (str): expression on which to terminate
    """
    # initialize stderr to None
    stderr = None

    # start logging
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

    return stderr


def finish_job_logger(joblogger, lfh, efh, stream_handler):
    """Close and remove file handlers.

    (to prevent OSError: [Errno 24] Too many open files)
    """
    # we need to close the FileHandlers to
    lfh.close()
    efh.close()
    joblogger.removeHandler(lfh)
    joblogger.removeHandler(efh)
    if stream_handler is not None:
        stream_handler.close()
        joblogger.removeHandler(stream_handler)


def log_through_print(logger, command):
    """Parse print output to logger.

    This can be used e.g. for printing a GP kernel or a pandas DataFrame.
    It works for all objects that implement a print method.

    Args:
        logger (Logger): logger to parse to print output to
        command (object): command/object which should be printed
    """
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    print(command)

    output = new_stdout.getvalue()
    split_data = output.splitlines()
    for current_line in split_data:
        logger.info(current_line)
    logger.info('')

    sys.stdout = old_stdout
