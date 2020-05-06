"""
Injector module supplies function to inject parameter values into a template text file.
"""


def inject(params, file_template, output_file):
    """ Injector function to insert parameters into file templates

    Args:
        params (dict):          dict with parameters to inject
        file_template (str):    file name including path to template
        output_file (str):      name of output file with injected parameters
    """

    with open(file_template, encoding='utf-8') as f:
        my_file = f.read()
    for name, value in params.items():
        if type(value).__name__ == 'ndarray':
            value = value[0]
        my_file = my_file.replace('{{{}}}'.format(name), str(value))

    with open(output_file, mode='w', encoding='utf-8') as f:
        f.write(my_file)
