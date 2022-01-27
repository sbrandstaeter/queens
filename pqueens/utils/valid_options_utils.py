"""Helper functions for valid options and switch analogy."""


def get_option(options_dict, desired_option, error_message=""):
    """Get option desired_option from the options_dict.

    The options_dict consists of the keys and their value. Note that the value can also be
    functions. In case the option is not found an error is raised.

    Args:
        options_dict (dict): Dictionary with valid options and their value
        desired_option (str): Desired method key

    Returns:
        value of the desired_option
    """
    if check_if_valid_option(list(options_dict.keys()), desired_option, error_message):
        return options_dict[desired_option]


def check_if_valid_option(valid_options, desired_option, error_message=""):
    """Check if the dersired option is in valid_options.

    Args:
        valid_options (lst): List of valid option keys
        desired_option (str): Key of desired option
        error_message (str, optional): Error message in case the desired option can not be found.

    Returns:
        True: if the desired option is in valid_options
    """
    if desired_option in valid_options:
        return True
    else:
        raise InvalidOptionError.construct_error_from_options(
            valid_options, desired_option, error_message
        )


class InvalidOptionError(Exception):
    """Custom error class for invalid options during QUEENS runs."""

    def __init__(self, message, valid_options=None, desired_option=None, additional_message=None):
        """Initialise InvalidOptionError.

        Args:
            message (str): Error message
            valid_options (lst): List of valid option keys
            desired_option (str): Key of desired option
            additional_message (str, optional): Additional message to pass. Defaults to None.
        """
        self.valid_options = valid_options
        self.desired_option = desired_option
        self.additional_message = additional_message
        self.message = message
        super().__init__(self.message)

    @classmethod
    def construct_error_from_options(cls, valid_options, desired_option, additional_message=""):
        """Construct invalid option error from the valid and desired options.

        Args:
            valid_options (lst): List of valid option keys
            desired_option (str): Key of desired option
            additional_message (str, optional): Additional message to pass. Defaults to None.

        Returns:
            InvalidOptionError
        """
        message = cls.construct_error_message(valid_options, desired_option, additional_message)
        return cls(message, valid_options, desired_option, additional_message)

    @staticmethod
    def construct_error_message(valid_options, desired_option, additional_message):
        """Construct error message based on the valid and desired options.

        Args:
            valid_options (lst): List of valid option keys
            desired_option (str): Key of desired option
            additional_message (str, optional): Additional message to pass. Defaults to None.

        Returns:
            message (str): Error message that is display once the error is raised
        """
        message = "\n" + additional_message
        message += f"\nInvalid option '{desired_option}'. Valid options are:\n" + ", ".join(
            sorted(valid_options)
        )
        return message
