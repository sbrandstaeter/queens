"""Command Line Interface utils collection."""
import argparse
import logging
import sys
from pathlib import Path

from pqueens.utils import ascii_art
from pqueens.utils.exceptions import CLIError
from pqueens.utils.injector import inject
from pqueens.utils.logger_settings import reset_logging, setup_cli_logging
from pqueens.utils.path_utils import PATH_TO_QUEENS
from pqueens.utils.pickle_utils import print_pickled_data
from pqueens.utils.print_utils import get_str_table
from pqueens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)


def cli_logging(func):
    """Decorator to create logger for CLI function.

    Args:
        func (function): Function that is to be decorated
    """

    def decorated_function(*args, **kwargs):
        setup_cli_logging()
        results = func(*args, **kwargs)
        reset_logging()

        # For CLI commands there should be no results, but just in case
        return results

    return decorated_function


@cli_logging
def inject_template_cli():
    """Use the injector of QUEENS."""
    ascii_art.print_crown(80)
    ascii_art.print_banner("Injector", 80)
    parser = argparse.ArgumentParser(
        description="QUEENS injection CLI for Jinja2 templates. The parameters '--name <value>'"
    )
    parser.add_argument(
        '--template',
        type=str,
        help="Jinja2 template to be injected.",
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Output path for the injected template.',
    )

    # These two are dummy arguments to indicate how to use this CLI
    parser.add_argument(
        '--paramter_1',
        type=str,
        default=None,
        help='Parameter 1 and value to be inject into the template.',
    )
    parser.add_argument(
        '--paramter_2',
        type=str,
        default=None,
        help='Parameter 2 and value to be inject into the template.',
    )

    path_arguments, injection_dict = parser.parse_known_args()

    template_path = Path(path_arguments.template)
    if path_arguments.output_path is None:
        output_path = template_path.with_name(
            template_path.stem + '_injected' + template_path.suffix
        )
    else:
        output_path = Path(path_arguments.output_path)

    _logger.info("Template: %s", template_path.resolve())
    _logger.info("Output path: %s", template_path.resolve())
    _logger.info(" ")

    # Get injection parameters
    injection_parser = argparse.ArgumentParser()

    # Add input parameters to inject
    for arg in injection_dict:
        if arg.find("--") > -1:
            injection_parser.add_argument(arg)

    # Create the dictionary
    injection_dict = vars(injection_parser.parse_args(injection_dict))
    _logger.info(get_str_table("Injection parameters", injection_dict))
    inject(injection_dict, template_path, output_path)

    _logger.info("Injection done, created file %s", output_path)


@cli_logging
def print_pickle_data_cli():
    """Print pickle data wrapper."""
    ascii_art.print_crown(60)
    ascii_art.print_banner("QUEENS", 60)
    args = sys.argv[1:]
    if len(args) == 0:
        _logger.info('No pickle file was provided!')
    else:
        file_path = args[0]
        print_pickled_data(Path(file_path))


def build_html_coverage_report():
    """Build html coverage report."""
    _logger.info('Build html coverage report...')

    pytest_command_string = (
        'pytest -m "unit_tests or integration_tests or integration_tests_baci" '
        '--cov --cov-report=html:html_coverage_report'
    )
    command_list = ["cd", str(PATH_TO_QUEENS), "&&", pytest_command_string]
    command_string = ' '.join(command_list)
    run_subprocess(command_string)


def remove_html_coverage_report():
    """Remove html coverage report files."""
    _logger.info('Remove html coverage report...')

    pytest_command_string = "rm -r html_coverage_report/; rm .coverage*"
    command_list = ["cd", str(PATH_TO_QUEENS), "&&", pytest_command_string]
    command_string = ' '.join(command_list)
    run_subprocess(command_string)


def str_to_bool(value):
    """Convert string to boolean for cli commands.

    Args:
        value (str): String to convert to a bool

    Returns:
        bool: Bool of the string
    """
    if isinstance(value, bool):
        return value

    false_options = ('false', 'f', '0', 'no', 'n')
    true_options = ('true', 't', '1', 'yes', 'y')
    if value.lower() in false_options:
        return False
    if value.lower() in true_options:
        return True
    raise CLIError(
        f"{value} is not a valid boolean value. Valid options are:\n"
        f"{', '.join(list(true_options+false_options))}"
    )


def get_cli_options(args):
    """Get input file path, output directory and debug from args.

    Args:
        args (list): cli arguments

    Returns:
        input_file (Path): Path object to input file
        output_dir (Path): Path object to the output directory
        debug (bool):      *True* if debug mode is to be used
    """
    parser = argparse.ArgumentParser(description="QUEENS")
    parser.add_argument(
        '--input', type=str, default=None, help='Input file in .json or .yaml/yml format.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory to write results to. The directory has to be created by the user!',
    )
    parser.add_argument('--debug', type=str_to_bool, default=False, help='Debug mode yes/no.')

    args = parser.parse_args(args)

    if args.input is None:
        raise CLIError("No input file was provided with option --input.")

    if args.output_dir is None:
        raise CLIError("No output directory was provided with option --output_dir.")

    debug = args.debug
    output_dir = Path(args.output_dir)
    input_file = Path(args.input)

    return input_file, output_dir, debug


@cli_logging
def print_greeting_message():
    """Print a greeting message and how to use QUEENS."""
    ascii_art.print_banner_and_description()
    ascii_art.print_centered_multiline("Welcome to the royal family!")
    _logger.info('\nTo use QUEENS run:\n')
    _logger.info('queens --input <inputfile> --output_dir <output_dir>\n')
    _logger.info('or\n')
    _logger.info('python -m pqueens.main --input <inputfile> --output_dir <output_dir>\n')
    _logger.info('or\n')
    _logger.info(
        'python path_to_queens/pqueens/main.py --input <inputfile> --output_dir <output_dir>\n'
    )
