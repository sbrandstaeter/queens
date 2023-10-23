"""Main module of QUEENS containing the high-level control routine."""
import logging
import sys
import time

import queens.global_settings
from queens.global_settings import GlobalSettings
from queens.utils.ascii_art import print_banner_and_description
from queens.utils.cli_utils import get_cli_options, print_greeting_message
from queens.utils.fcc_utils import from_config_create_iterator
from queens.utils.io_utils import load_input_file

_logger = logging.getLogger(__name__)


def run(input_file, output_dir, debug=False):
    """Do a QUEENS run.

    Args:
        input_file (Path): Path object to the input file
        output_dir (Path): Path object to the output directory
        debug (bool): True if debug mode is to be used
    """
    start_time_input = time.time()

    # read input and create config
    config = load_input_file(input_file)

    experiment_name = config.pop('experiment_name')

    with GlobalSettings(experiment_name=experiment_name, output_dir=output_dir, debug=debug):
        # create iterator
        my_iterator = from_config_create_iterator(config)

        end_time_input = time.time()

        _logger.info("")
        _logger.info("Time for INPUT: %s s", end_time_input - start_time_input)
        _logger.info("")

        # perform analysis
        run_iterator(my_iterator)


def run_iterator(iterator):
    """Run the main queens iterator.

    Args:
        iterator (Iterator): Main queens iterator
    """
    print_banner_and_description()
    queens.global_settings.GLOBAL_SETTINGS.print_git_information()

    start_time_calc = time.time()

    _logger.info("")
    _logger.info("Starting Analysis...")
    _logger.info("")

    try:
        iterator.run()
    except Exception as exception:
        _logger.exception(exception)
        queens.global_settings.GLOBAL_SETTINGS.__exit__(None, None, None)
        # TODO: Write iterator in pickle file
        raise exception

    end_time_calc = time.time()
    _logger.info("")
    _logger.info("Time for CALCULATION: %s s", end_time_calc - start_time_calc)
    _logger.info("")


def main():
    """Main function."""
    # the first argument is the file name
    args = sys.argv[1:]

    if len(args) > 0:
        # do QUEENS run
        input_file_path, output_dir, debug = get_cli_options(args)
        run(input_file_path, output_dir, debug)
    else:
        # print some infos
        print_greeting_message()


if __name__ == '__main__':
    sys.exit(main())
