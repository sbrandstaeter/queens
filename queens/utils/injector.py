"""Injector module.

The module supplies functions to inject parameter values into a template
text file.
"""

from pathlib import Path

from jinja2 import Environment, Template, meta

from queens.utils.exceptions import InjectionError
from queens.utils.io_utils import read_file


def render_template(params, template, strict=True):
    """Function to insert parameters into a template.

    Args:
        params (dict): Dict with parameters to inject
        template (str): Template file as string
        output_file (str, Path): Name of output file with injected parameters
        strict (bool): Raises exception if mismatch between provided and required parameters

    Returns:
        str: injected template
    """
    if strict:
        # Get the required parameters
        template_obj = Environment().parse(template)
        required_parameters_in_template = meta.find_undeclared_variables(template_obj)

        # Get the parameters to be injected
        provided_parameters = set(params.keys())

        # In case of mismatch raise Exception
        if required_parameters_in_template.symmetric_difference(provided_parameters):
            raise InjectionError.construct_error(
                required_parameters_in_template, provided_parameters
            )
    return Template(template).render(**params)


def inject_in_template(params, template, output_file, strict=True):
    """Function to insert parameters into file template and write to file.

    Args:
        params (dict): Dict with parameters to inject
        template (str): Template (str)
        output_file (str, Path):    Name of output file with injected parameters
        strict (bool): Raises exception if mismatch between provided and required parameters
    """
    injected_template = render_template(params, template, strict)
    Path(output_file).write_text(injected_template, encoding="utf-8")


def inject(params, template_path, output_file, strict=True):
    """Function to insert parameters into file template and write to file.

    Args:
        params (dict): Dict with parameters to inject
        template_path (str, Path): Path to template
        output_file (str, Path):    Name of output file with injected parameters
        strict (bool): Raises exception if mismatch between provided and required parameters
    """
    template = read_file(template_path)
    inject_in_template(params, template, output_file, strict)
