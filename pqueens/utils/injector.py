"""Injector module.

The module supplies functions to inject parameter values into a template
text file.
"""

from jinja2 import Template


def inject(params, file_template, output_file):
    """Function to insert parameters into file templates.

    Args:
        params (dict):          Dict with parameters to inject
        file_template (Path):    File name including path to template
        output_file (Path):      Name of output file with injected parameters
    """
    template = Template(file_template.read_text(encoding='utf-8'))
    output_file.write_text(template.render(**params), encoding='utf-8')
