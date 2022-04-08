"""Print utils."""


def get_str_table(name, print_dict):
    """Function to get table to be used in `__str__` methods.

    Args:
        name (str): object name
        print_dict (dict): dict containing labels and values to print

    Returns:
        str: table to print
    """
    seperator_line = "-" * 60 + "\n"

    string = "\n" + seperator_line
    string += name + "\n"
    string += seperator_line
    for item in print_dict.items():
        string += f"{item[0]}: {item[1]}\n"
    string += seperator_line + "\n"
    return string
