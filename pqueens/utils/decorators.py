"""
Collection of decorators.
"""


def safe_mongodb_operation(fun):
    """
    Method decorator in order to process QUEENS MongoDB wrapper methods safely.
    Do **not** use this decorator on functions with @classmethod decorators! (As we extract the 
    self argument which does not exist for @classmethod functions)

    The safe procedure consists of:
        - Catching and logging exceptions
        - Retrying the method call up to `max_number_of_attempts` times

    Args:
        fun (method): MongoDB method trying to access the database

    Returns:
        wrapper (obj): function that is called instead of `fun`
    """

    def wrapper(*args, **kwargs):
        """
        Function in which the method `fun` is wrapped in. In here exceptions are caught and the
        operation is retried up to `self.max_number_of_attempts` times
        
        Returns:
            A safe `fun` call.
        """
        # Get the self attribute
        self = args[0]

        # Repeat function call if failed
        for attempt in range(1, self.max_number_of_attempts + 1):
            try:
                return fun(*args, **kwargs)
                break
            except Exception as error:
                if attempt <= self.max_number_of_attempts:
                    _logger.warning(
                        f"Function {fun.__name__}: Attempt number {attempt}/"
                        f"{self.max_number_of_attempts} to '{self.db_address}' failed!"
                    )
                else:
                    raise ValueError(
                        f"Function {fun.__name__} failed with on '{self.db_address}' after"
                        f" {self.max_number_of_attempts} attempts!"
                    ) from error

    return wrapper
