"""Print utils."""


def get_str_table(name, print_dict):
    """Function to get table to be used in *__str__* methods.

    Args:
        name (str): Object name
        print_dict (dict): Dict containing labels and values to print

    Returns:
        str: Table to print
    """
    # lines for the table
    lines = [name]
    for key, item in print_dict.items():
        lines.append(f"  {key}: {item}")

    # find max width and create sepreators
    seperator_width = max(max(len(l) for l in lines), 63)
    if seperator_width % 2 == 0:
        # odd width just look better
        seperator_width += 1
    main_seperator_line = "+" + "-" * seperator_width + "+"
    soft_separator_line = "|" + "- " * (seperator_width // 2) + "-|"

    # Create table string
    string = main_seperator_line + "\n"
    for i, line in enumerate(lines):
        if i == 0:
            string += "|" + line.center(seperator_width) + "|\n"
            string += soft_separator_line + "\n"
        else:
            string += "|" + line.ljust(seperator_width) + "|\n"
    string += main_seperator_line + "\n"
    return string
