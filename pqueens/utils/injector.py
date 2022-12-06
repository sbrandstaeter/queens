"""Injector module.

the module supplies functions to inject parameter values into a template
text file.
"""

import json


def inject(params, file_template, output_file):
    """Function to insert parameters into file templates.

    Args:
        params (dict):          dict with parameters to inject
        file_template (Path):    file name including path to template
        output_file (Path):      name of output file with injected parameters
    """
    with open(file_template, encoding='utf-8') as f:
        my_file = f.read()
    for name, value in params.items():
        my_file = my_file.replace(f"{{{name}}}", str(value))

    with open(output_file, mode='w', encoding='utf-8') as f:
        f.write(my_file)


def inject_remote(path_to_params_json, file_template, output_file):
    """Function to insert parameters into file templates on remote machine.

    Args:
        path_to_params_json (str):  path to JSON file containing dict to inject
        file_template (str):        file name including path to template
        output_file (str):          name of output file with injected parameters
    """
    # load parameter JSON file into parameter dictionary
    with open(path_to_params_json, mode='r', encoding='utf-8') as f:
        params = json.load(f)

    # call standard injector function
    inject(params, file_template, output_file)


if __name__ == "__main__":

    import sys

    sys.exit(inject_remote(sys.argv[1], sys.argv[2], sys.argv[3]))
