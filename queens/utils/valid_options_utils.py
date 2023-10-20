"""Helper functions for valid options and switch analogy."""
from queens.utils.exceptions import InvalidOptionError


def get_option(options_dict, desired_option, error_message=""):
    """Get option *desired_option* from *options_dict*.

    The *options_dict* consists of the keys and their values. Note that the value can also be
    functions. In case the option is not found an error is raised.

    Args:
        options_dict (dict): Dictionary with valid options and their value
        desired_option (str): Desired method key
        error_message: TODO_doc

    Returns:
        Value of the *desired_option*
    """
    check_if_valid_options(list(options_dict.keys()), desired_option, error_message)
    return options_dict[desired_option]


def check_if_valid_options(valid_options, desired_options, error_message=""):
    """Check if the desired option(s) is/are in valid_options.

    Raises InvalidOptionError if invalid options are present.

    Args:
        valid_options (lst,dict): List of valid option keys or dict with valid options as keys
        desired_options (str, lst(str), dict): Key(s) of desired options
        error_message (str, optional): Error message in case the desired option can not be found
    """
    desired_options_set = set(desired_options)
    if isinstance(desired_options, str):
        desired_options_set = {desired_options}

    # Set of options that are not valid
    invalid_options = (desired_options_set ^ set(valid_options)) - set(valid_options)

    if invalid_options:
        raise InvalidOptionError.construct_error_from_options(
            valid_options, ", ".join(desired_options_set), error_message
        )
