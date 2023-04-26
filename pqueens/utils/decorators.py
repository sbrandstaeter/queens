"""Collection of decorators."""
import logging
from functools import wraps
from time import sleep

_logger = logging.getLogger(__name__)


def safe_operation(function, max_number_of_attempts=10, waiting_time=0):
    """Method decorator in order to process database methods safely.

    The safe procedure consists of:
        - Catching and logging exceptions
        - Waiting *waiting_time*
        - Retrying the method call up to *max_number_of_attempts* times

    Args:
        function (function): Function trying to access the database
        max_number_of_attempts: TODO_doc
        waiting_time: TODO_doc

    Returns:
        wrapper (obj): Function that is called instead of *function*
    """

    @wraps(function)
    def wrapper(self, *args, **kwargs):
        """Function in which the method `function` is wrapped in.

        In here exceptions are caught and the operation is retried up to
        `max_number_of_attempts` times.

        Returns:
            A safe `function` call.
        """
        # Repeat function call if failed
        for attempt in range(1, max_number_of_attempts + 1):
            try:
                return function(self, *args, **kwargs)
            except Exception as error:
                function_name = function.__module__ + ":" + function.__name__
                if attempt < max_number_of_attempts:
                    _logger.warning(
                        "Function %s: Attempt number %s/%s",
                        function_name,
                        attempt,
                        max_number_of_attempts,
                    )
                    sleep(waiting_time)
                else:
                    # Raise the same error type as the original exception did
                    raise type(error)(
                        f"Function {function_name} failed after {max_number_of_attempts} attempts"
                        f" for the database\n{str(self)}"
                    ) from error

    return wrapper
