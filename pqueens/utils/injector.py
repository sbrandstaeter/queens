"""Injector module.

The module supplies functions to inject parameter values into a template
text file.
"""

from pathlib import Path

from jinja2 import Template


def read_file(file_path):
    """Function to read in a file.

    Args:
        file_path (str, Path): Path to file
    Returns:
        file (str): read in file as string
    """
    file = Path(file_path).read_text(encoding='utf-8')
    return file


def inject_in_template(params, template, output_file):
    """Function to insert parameters into a template.

    Args:
        params (dict):          Dict with parameters to inject
        template (str):         Template file as string
        output_file (str):      Name of output file with injected parameters
    """
    Path(output_file).write_text(Template(template).render(**params), encoding='utf-8')


def inject(params, file_template, output_file):
    """Function to insert parameters into file templates.

    Args:
        params (dict):          Dict with parameters to inject
        file_template (str):    File name including path to template
        output_file (str):      Name of output file with injected parameters
    """
    template = read_file(file_template)
    inject_in_template(params, template, output_file)
