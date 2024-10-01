"""Injector module.

The module supplies functions to inject parameter values into a template
text file.
"""

from pathlib import Path

from jinja2 import Environment, StrictUndefined, Undefined

from queens.utils.io_utils import read_file


def render_template(params, template, strict=True):
    """Function to insert parameters into a template.

    Args:
        params (dict): Dict with parameters to inject
        template (str): Template file as string
        strict (bool): Raises exception if required parameters from the template are missing

    Returns:
        str: injected template
    """
    undefined = StrictUndefined if strict else Undefined

    environment = Environment(undefined=undefined).from_string(template)
    return environment.render(**params)


def inject_in_template(params, template, output_file, strict=True):
    """Function to insert parameters into file template and write to file.

    Args:
        params (dict): Dict with parameters to inject
        template (str): Template (str)
        output_file (str, Path): Name of output file with injected parameters
        strict (bool): Raises exception if mismatch between provided and required parameters
    """
    injected_template = render_template(params, template, strict)
    Path(output_file).write_text(injected_template, encoding="utf-8")


def inject(params, template_path, output_file, strict=True):
    """Function to insert parameters into file template and write to file.

    Args:
        params (dict): Dict with parameters to inject
        template_path (str, Path): Path to template
        output_file (str, Path): Name of output file with injected parameters
        strict (bool): Raises exception if mismatch between provided and required parameters
    """
    template = read_file(template_path)
    inject_in_template(params, template, output_file, strict)
