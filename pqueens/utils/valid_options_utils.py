"""Helper functions for valid options and switch analogy."""
from pqueens.utils.exceptions import InvalidOptionError


def get_option(options_dict, desired_option, error_message=""):
    """Get option desired_option from the options_dict.

    The options_dict consists of the keys and their values. Note that the value can also be
    functions. In case the option is not found an error is raised.

    Args:
        options_dict (dict): Dictionary with valid options and their value
        desired_option (str): Desired method key

    Returns:
        Value of the desired_option
    """
    check_if_valid_option(list(options_dict.keys()), desired_option, error_message)
    return options_dict[desired_option]


def check_if_valid_option(valid_options, desired_option, error_message=""):
    """Check if the desired option is in valid_options.

    Args:
        valid_options (lst): List of valid option keys
        desired_option (str): Key of desired option
        error_message (str, optional): Error message in case the desired option can not be found.

    Returns:
        True: if the desired option is in valid_options
    """
    if desired_option in valid_options:
        return True

    raise InvalidOptionError.construct_error_from_options(
        valid_options, desired_option, error_message
    )


def get_valid_options(valid_options, desired_options, error_message=""):
    """Get multiple desired options from a list of valid options.

    Args:
        valid_options (lst): List of valid option keys
        desired_options (lst): List of desired options
        error_message (str, optional): Error message in case the desired option can not be found.

    Returns:
        list: List with the unique desired and valid options
    """
    unvalid_options = list((set(desired_options) ^ set(valid_options)) - set(valid_options))
    if unvalid_options:
        raise InvalidOptionError.construct_error_from_options(
            valid_options, ", ".join(desired_options), error_message
        )
    # remove eventual duplicates
    return list(set(desired_options))
