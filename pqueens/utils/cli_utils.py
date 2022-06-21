"""Command Line Interface utils collection."""
import sys

from pqueens.utils import ascii_art
from pqueens.utils.manage_singularity import create_singularity_image
from pqueens.utils.path_utils import PATH_TO_QUEENS
from pqueens.utils.pickle_utils import print_pickled_data
from pqueens.utils.run_subprocess import run_subprocess


def build_singularity_cli():
    """Build singularity image CLI wrapper."""
    ascii_art.print_crown(75)
    ascii_art.print_banner("SINBUILD", 75)
    print("Singularity image builder for QUEENS runs".center(75))

    print("\n\nBuilding a singularity image! This might take some time ...")
    try:
        create_singularity_image()
        print("Done!")
    except Exception as cli_singularity_error:
        print("Building singularity failed!\n\n")
        raise cli_singularity_error


def print_pickle_data_cli():
    """Print pickle data wrapper."""
    ascii_art.print_crown(60)
    ascii_art.print_banner("QUEENS", 60)
    args = sys.argv[1:]
    if len(args) == 0:
        print("No pickle file was provided!")
    else:
        file_path = args[0]
        print_pickled_data(file_path)


def build_html_coverage_report():
    """Build html coverage report."""
    print("Build html coverage report...")

    pytest_command_string = (
        'pytest -m "unit_tests or integration_tests or integration_tests_baci" '
        '--cov --cov-report=html:html_coverage_report'
    )
    command_list = ["cd", str(PATH_TO_QUEENS), "&&", pytest_command_string]
    command_string = ' '.join(command_list)
    run_subprocess(command_string)


def remove_html_coverage_report():
    """Remove html coverage report files."""
    print("Remove html coverage report...")

    pytest_command_string = "rm -r html_coverage_report/; rm .coverage*"
    command_list = ["cd", str(PATH_TO_QUEENS), "&&", pytest_command_string]
    command_string = ' '.join(command_list)
    run_subprocess(command_string)
