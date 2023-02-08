"""Logging in QUEENS."""

import io
import logging
import re
import sys
import time


class LogFilter(logging.Filter):
    """Filters (lets through) all messages with level <= LEVEL.

    Attributes:
        level: TODO_doc
    """

    def __init__(self, level):
        """Initiatlize the logging filter.

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
            LogRecord obj: Filter logging record
        """
        return record.levelno <= self.level


class NewLineFormatter(logging.Formatter):
    """Formatter splitting multiline messages into single line messages.

    A logged message that consists of more than one line - contains a new line char - is split
    into multiple single line messages that all have the same format.
    Without this the overall format of the logging is broken for by multiline messages.
    """

    def __init__(self, fmt, datefmt=None):
        """Initialize the NewLineFormatter.

        Args:
            fmt (str): Use the specified format string for the handler.
            datefmt (str): Use the specified date/time format, as accepted by time.strftime().
        """
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        """Override format function.

        Args:
            record (LogRecord obj): Logging record object
        Returns:
            formatted_message (str): logged message in supplied format split into single lines
        """
        formatted_message = logging.Formatter.format(self, record)

        if record.message != "":
            parts = formatted_message.split(record.message)
            formatted_message = formatted_message.replace('\n', '\n' + parts[0])

        return formatted_message


def setup_basic_logging(output_dir, experiment_name):
    """Setup basic logging.

    Args:
        output_dir (Path): Output directory where to save the log-file
        experiment_name (str): Experiment name used as file name for the log-file
    """
    file_level_min = logging.DEBUG
    console_level_min = logging.INFO

    library_logger = logging.getLogger("pqueens")
    # call setLevel() for basic initialisation (this is needed not sure why)
    library_logger.setLevel(logging.DEBUG)

    # set up logging to file
    logging_file_path = output_dir / f"{experiment_name}.log"
    file_handler = logging.FileHandler(logging_file_path, mode="w")
    file_formatter = NewLineFormatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(file_level_min)
    library_logger.addHandler(file_handler)

    # a plain, minimal formatter for streamhandlers
    stream_formatter = NewLineFormatter('%(message)s')

    # set up logging to stdout
    console_stdout = logging.StreamHandler(stream=sys.stdout)
    # messages lower than and including WARNING go to stdout
    log_filter = LogFilter(logging.WARNING)
    console_stdout.addFilter(log_filter)
    console_stdout.setLevel(console_level_min)
    console_stdout.setFormatter(stream_formatter)
    library_logger.addHandler(console_stdout)

    # set up logging to stderr
    console_stderr = logging.StreamHandler(stream=sys.stderr)
    # messages >= ERROR or messages >= CONSOLE_LEVEL_MIN if CONSOLE_LEVEL_MIN > ERROR go to stderr
    console_stderr.setLevel(max(console_level_min, logging.ERROR))
    console_stderr.setFormatter(stream_formatter)
    library_logger.addHandler(console_stderr)

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
    formatter = NewLineFormatter('%(name)-12s: %(levelname)-8s %(message)s')
    console_stdout.setFormatter(formatter)
    console_stderr.setFormatter(formatter)

    # add the handlers to the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(console_stdout)
    root_logger.addHandler(console_stderr)


def get_job_logger(logger_name, log_file, error_file, streaming, propagate=False):
    """Setup job logging and get job logger.

    Args:
        logger_name (str): Logger name
        log_file (path): Path to log file
        error_file (path): Path to error file
        streaming (bool): Flag for additional streaming to given stream
        propagate (bool): Flag for propagation of stream (default: *False*)

    Returns:
        TODO_doc
    """
    # get job logger
    joblogger = logging.getLogger(logger_name)

    # define formatter
    formatter = NewLineFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
        process (obj): Subprocess object
        joblogger (obj): Job logger object
        terminate_expr (str): Expression on which to terminate

    Returns:
        TODO_doc
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

    Args:
        joblogger: TODO_doc
        lfh: TODO_doc
        efh: TODO_doc
        stream_handler: TODO_doc
    """
    # we need to close the FileHandlers to
    lfh.close()
    efh.close()
    joblogger.removeHandler(lfh)
    joblogger.removeHandler(efh)
    if stream_handler is not None:
        stream_handler.close()
        joblogger.removeHandler(stream_handler)
