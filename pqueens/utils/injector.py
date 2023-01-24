"""Injector module.

The module supplies functions to inject parameter values into a template
text file.
"""

from pathlib import Path

from jinja2 import Template


def inject(params, file_template, output_file):
    """Function to insert parameters into file templates.

    Args:
        params (dict):          Dict with parameters to inject
        file_template (str):    File name including path to template
        output_file (str):      Name of output file with injected parameters
    """
    with Path(file_template).open(encoding='utf-8') as f:
        template = Template(f.read())

    with Path(output_file).open(mode='w', encoding='utf-8') as f:
        f.write(template.render(**params))
