"""Main module of QUEENS containing the high-level control routine.

Handles input parsing. Controls and runs the analysis.
"""
import argparse
import os
import pathlib
import sys
import time
from collections import OrderedDict

import pqueens.database.database as DB_module
import pqueens.parameters.parameters as parameters_module
from pqueens.external_geometry import from_config_create_external_geometry
from pqueens.iterators import from_config_create_iterator
from pqueens.utils import ascii_art
from pqueens.utils.logger_settings import setup_basic_logging

try:
    import simplejson as json
except ImportError:
    import json


def main(args=None):
    """Main function of QUEENS.

    controls and runs the analysis.

    Args:
        args (list): list of arguments to be parsed
    """
    ascii_art.print_crown()
    ascii_art.print_banner()
    description = """
    A general purpose framework for Uncertainty Quantification,
    Physics-Informed Machine Learning, Bayesian Optimization,
    Inverse Problems and Simulation Analytics
    """
    ascii_art.print_centered_multiline(description)

    if not args:
        args = sys.argv[1:]

    # read input
    start_time_input = time.time()
    options = get_options(args)

    # stop here if no options are provided
    if options is None:
        return

    setup_basic_logging(
        pathlib.Path(options["global_settings"]["output_dir"]),
        options["global_settings"]["experiment_name"],
    )
    DB_module.from_config_create_database(options)

    with DB_module.database as db:

        pre_processer = from_config_create_external_geometry(options, 'pre_processing')
        if pre_processer:
            pre_processer.main_run()
            pre_processer.write_random_fields_to_dat()

        parameters_module.from_config_create_parameters(options, pre_processer)
        # build iterator
        my_iterator = from_config_create_iterator(options)

        end_time_input = time.time()

        print("")
        print(f"Time for INPUT: {end_time_input - start_time_input} s")
        print("")

        start_time_calc = time.time()

        print("")
        print("Starting Analysis...")
        print("")

        # perform analysis
        my_iterator.run()

    end_time_calc = time.time()
    print("")
    print(f"Time for CALCULATION: {end_time_calc - start_time_calc} s")
    print("")


def get_options(args):
    """Parse options from command line and input file.

    Args:
        args (list): list of arguments to be parsed

    Returns:
        dict: parsed options in a dictionary
    """
    parser = argparse.ArgumentParser(description="QUEENS")
    parser.add_argument('--input', type=str, default=None, help='Input file in .json format.')
    parser.add_argument(
        '--output_dir', type=str, default=None, help='Output directory to write results to.'
    )
    parser.add_argument('--debug', type=str, default='no', help='debug mode yes/no')

    args = parser.parse_args(args)

    # if no options are provided print a greeting message
    if args.input is None and args.output_dir is None:
        ascii_art.print_centered_multiline("Welcome to the royal family!")
        print("\nTo use QUEENS run:\n")
        print("queens --input <inputfile> --output_dir <output_dir>\n")
        print("or\n")
        print("python -m pqueens.main --input <inputfile> --output_dir <output_dir>\n")
        print("or\n")
        print(
            "python path_to_queens/pqueens/main.py --input <inputfile> --output_dir <output_dir>\n"
        )
        return None

    if args.input is None:
        raise Exception("No json input file was provided.")

    if args.output_dir is None:
        raise Exception("No output directory was given.")

    output_dir = os.path.realpath(os.path.expanduser(args.output_dir))
    if not os.path.isdir(output_dir):
        raise Exception("Output directory does not exist.")

    input_file = os.path.realpath(os.path.expanduser(args.input))
    try:
        with open(input_file, 'r') as f:
            options = json.load(f, object_pairs_hook=OrderedDict)
    except Exception as exception:
        raise FileNotFoundError("config.json did not load properly.") from exception

    if args.debug == 'yes':
        debug = True
    elif args.debug == 'no':
        debug = False
    else:
        print('Warning input flag not set correctly not showing debug information')
        debug = False

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


if __name__ == '__main__':
    sys.exit(main())
