"""Main module of QUEENS containing the high-level control routine."""
import logging
import sys
import time
from pathlib import Path

from pqueens.schedulers.scheduler import SHUTDOWN_CLIENTS
from pqueens.utils.ascii_art import print_banner_and_description
from pqueens.utils.cli_utils import get_cli_options, print_greeting_message
from pqueens.utils.fcc_utils import from_config_create_iterator
from pqueens.utils.io_utils import load_input_file
from pqueens.utils.logger_settings import setup_basic_logging

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
    config = get_config_dict(input_file, output_dir, debug)

    # set up logging
    log_file_path = Path(config["global_settings"]["output_dir"]) / (
        f'{config["global_settings"]["experiment_name"]}.log'
    )
    setup_basic_logging(
        log_file_path=log_file_path,
        debug=debug,
    )

    print_banner_and_description()

    # create iterator
    my_iterator = from_config_create_iterator(config)

    end_time_input = time.time()

    _logger.info("")
    _logger.info("Time for INPUT: %s s", end_time_input - start_time_input)
    _logger.info("")

    start_time_calc = time.time()

    _logger.info("")
    _logger.info("Starting Analysis...")
    _logger.info("")

    # perform analysis
    my_iterator.run()

    end_time_calc = time.time()
    _logger.info("")
    _logger.info("Time for CALCULATION: %s s", end_time_calc - start_time_calc)
    _logger.info("")
    for shutdown_client in SHUTDOWN_CLIENTS.copy():
        SHUTDOWN_CLIENTS.remove(shutdown_client)
        shutdown_client()


def get_config_dict(input_file, output_dir, debug=False):
    """Create QUEENS run config from input file and output dir.

    Args:
        input_file (Path): Path object to the input file
        output_dir (Path): Path object to the output directory
        debug (bool): True if debug mode is to be used

    Returns:
        dict: config dict
    """
    if not Path(output_dir).is_dir():
        raise FileNotFoundError(f"Output directory {output_dir} does not exist.")

    options = load_input_file(input_file)

    options["debug"] = debug
    options["input_file"] = input_file

    # move some parameters into a global settings dict to be passed to e.g.
    # iterators facilitating input output stuff
    global_settings = {}
    global_settings["output_dir"] = output_dir
    global_settings["experiment_name"] = options["experiment_name"]

    # remove experiment_name field from options dict
    options["global_settings"] = global_settings

    # remove experiment_name field from options dict make copy first
    final_options = dict(options)
    del final_options["experiment_name"]

    return final_options


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
