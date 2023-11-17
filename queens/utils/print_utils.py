"""Print utils."""


DEFAULT_OUTPUT_WIDTH = 81


def get_str_table(name, print_dict):
    """Function to get table to be used in *__str__* methods.

    Args:
        name (str): Object name
        print_dict (dict): Dict containing labels and values to print

    Returns:
        str: Table to print
    """
    column_name = [str(k) for k in print_dict.keys()]
    column_value = [repr(v).replace("\n", " ") for v in print_dict.values()]
    column_width_name = max(len(s) for s in column_name)
    column_width_value = max(len(s) for s in column_value)

    data_template = f"{{:<{column_width_name}}} : {{:<{column_width_value}}}"

    # find max width and create seperators
    seperator_width = max(
        max(len(data_template.format("", "")), len(name)) + 4, DEFAULT_OUTPUT_WIDTH
    )
    line_template = f"| {{:{seperator_width-4}}} |\n"
    main_seperator_line = "+" + "-" * (seperator_width - 2) + "+\n"
    soft_separator_line = (
        "|" + "- " * ((seperator_width - 2) // 2) + "-" * (seperator_width % 2) + "|\n"
    )

    # Create table string
    string = "\n" + main_seperator_line
    string += f"| {{:^{seperator_width-4}}} |\n".format(name)
    string += soft_separator_line
    for field_name, value in zip(column_name, column_value):
        content = data_template.format(field_name, value)
        string += line_template.format(content)
    string += main_seperator_line
    return string
