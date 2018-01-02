
# standard imports
import argparse
import os
from collections import OrderedDict
try:
    import simplejson as json
except ImportError:
    import json

# queens imports
from pqueens.iterators.iterator import Iterator

def get_options():
    """ Parse options from command line and input file """

    parser = argparse.ArgumentParser(description="QUEENS")
    parser.add_argument('--input', type=str, default='input.json',
                        help='Input file in .json format.')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory to write resutls to.')
    parser.add_argument('--debug', type=str, default='no',
                        help='debug mode yes/no')

    args = parser.parse_args()

    input_file = os.path.realpath(os.path.expanduser(args.input))
    try:
        with open(input_file, 'r') as f:
            options = json.load(f, object_pairs_hook=OrderedDict)
    except:
        raise Exception("config.json did not load properly.")

    if args.output_dir is None:
        raise Exception("No output directory was given.")

    output_dir = os.path.realpath(os.path.expanduser(args.output_dir))
    if not os.path.isdir(output_dir):
        raise Exception("Output directory was not set propertly.")

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
    options["output_dir"] = output_dir

    return  options

def main():
    """ Run analysis """
    options = get_options()

    # build iterator
    my_iterator = Iterator.from_config_create_iterator(options)

    # perform analysis
    my_iterator.run()

if __name__ == '__main__':
    main()
