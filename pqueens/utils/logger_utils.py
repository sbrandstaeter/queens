"""Logging in QUEENS"""


def log_multiline_string(logger, string):
    """Log multiline string line by line.

    Split a multiline string into lines and uses logger.info to generate output. This might
    enhance readability as it is in line with formatting of logging.

    Args:
        logger (Logger): logger to parse string to
        string (str): string which should be printed
    """

    for line in string.splitlines():
        logger.info(line)
