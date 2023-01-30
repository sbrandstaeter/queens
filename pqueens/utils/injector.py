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
    template = Template(Path(file_template).read_text(encoding='utf-8'))
    Path(output_file).write_text(template.render(**params), encoding='utf-8')
