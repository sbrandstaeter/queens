"""Utils handling user inputs."""

import logging
import signal

_logger = logging.getLogger(__name__)


def interrupted():
    """Interruption that is called when input times out."""
    raise ValueError("No user input within time limit.")


# initialize the interuptions signal
signal.signal(signal.SIGALRM, interrupted)


def request_user_input(default, timeout):
    """Request an input from the user.

    Args:
        default (string): Default value returned in case of timeout
        timeout (float): Time until interruption is called and default value returned (in seconds)

    Returns:
        string: ther default string or the string supplied by user input
    """
    try:
        user_input = input()
        return user_input
    except TypeError:
        # timeout
        _logger.info(
            "\nNo user input within time limit of %s s.\n Returning default value: %s \n",
            timeout,
            default,
        )
        return default


def request_user_input_with_default_and_timeout(default, timeout):
    """TODO_doc: add a one-line explanation.

    Wrapped around the user input request that manages the interruption
    after timeout seconds.

    Args:
        default (string): Default value returned in case of timeout
        timeout (float): Time until interruption is called and default value returned (in seconds)

    Returns:
        user_input: TODO_doc
    """
    # set alarm
    signal.alarm(timeout)
    user_input = request_user_input(default=default, timeout=timeout)
    # disable the alarm after success
    signal.alarm(0)
    return user_input
