
# standard imports
import sys
import argparse
import os
import pyfiglet
from collections import OrderedDict
try:
    import simplejson as json
except ImportError:
    import json

# queens imports
from pqueens.iterators.iterator import Iterator


def main(args):
    """ Run analysis """

    crown= """
                                *
                              * | *
                             * \|/ *
                        * * * \|O|/ * * *
                         \o\o\o|O|o/o/o/
                         (<><><>O<><><>)
                          '==========='
    """
    print(crown)
    result = pyfiglet.figlet_format("QUEENS", font = "banner3-D" )
    print(result)
    result ="""
 A general purpose framework for Uncertainty Quantification,
 Physics-Informed Machine Learning,
 Bayesian Optimization, Inverse Problems and Simulation Analytics
"""
    print(result)
    print("")
    print("Starting Analysis...")
    print("")
    options = get_options(args)

    # build iterator
    my_iterator = Iterator.from_config_create_iterator(options)

    # perform analysis
    my_iterator.run()


def get_options(args):
    """ Parse options from command line and input file """

    parser = argparse.ArgumentParser(description="QUEENS")
    parser.add_argument('--input', type=str, default='input.json',
                        help='Input file in .json format.')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory to write resutls to.')
    parser.add_argument('--debug', type=str, default='no',
                        help='debug mode yes/no')

    args = parser.parse_args(args)

    input_file = os.path.realpath(os.path.expanduser(args.input))
    try:
        with open(input_file, 'r') as f:
            options = json.load(f, object_pairs_hook=OrderedDict)
    except:
        raise FileNotFoundError("config.json did not load properly.")

    if args.output_dir is None:
        raise Exception("No output directory was given.")

    output_dir = os.path.realpath(os.path.expanduser(args.output_dir))
    if not os.path.isdir(output_dir):
        raise Exception("Output directory does not exist.")

    if args.debug == 'yes':
        debug = True
    elif args.debug == 'no':
        debug = False
    else:
        print('Warning input flag not set correctly not showing debug'
              ' information')
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
    sys.exit(main(sys.argv[1:]))
