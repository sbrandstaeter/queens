import logging
import sys


class LogFilter(logging.Filter):
    """Filters (lets through) all messages with level <= LEVEL"""
    def __init__(self, level):
        super().__init__()
        self.level = level

    def filter(self, record):
        return record.levelno <= self.level


def setup_logging(output_dir, experiment_name):
    """
    Setup basic logging

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
