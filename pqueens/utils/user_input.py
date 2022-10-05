import signal
import logging

_logger = logging.getLogger(__name__)


def interrupted():
    """Interpution that is called when input times out."""
    raise Exception(f"No user input within time limit.")


# initialize the interuptions signal
signal.signal(signal.SIGALRM, interrupted)


def request_user_input(default, timeout):
    """Request an input from the user
    Args:
        default: (string) default value returned in case of timeout
        timeout: (float) time until interuption is called and default value returen (in seconds)

    Returns: (string) either default string or the string supplied by user input
    """
    try:
        user_input = input()
        return user_input
    except:
        # timeout
        _logger.info(
            "\nNo user input within time limit of %s s.\n"
            "Returning default value: %s \n", timeout, default
        )
        return default


def request_user_input_with_default_and_timeout(default, timeout):
    """Wrappe around the user input request that manages the interuption after
    timeout seconds.

    Args:
        default: (string) default value returned in case of timeout
        timeout: (float) time until interuption is called and default value returen (in seconds)

    Returns:
    """
    # set alarm
    signal.alarm(timeout)
    user_input = request_user_input(default=default, timeout=timeout)
    # disable the alarm after success
    signal.alarm(0)
    return user_input
