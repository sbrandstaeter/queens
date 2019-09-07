import argparse
import os
from pqueens.drivers.driver import Driver
import sys
from collections import OrderedDict
try:
    import simplejson as json
except ImportError:
    import json

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", help="specify the job_id so that correct simulation settings can be loaded from the DB", type=int)
    parser.add_argument("--batch", help="specify the batch number of the simulation",type=int)
    parser.add_argument("--port", help="port number chosen for port-forwarding",type=str)
    parser.add_argument("--path_json", help="system path to temporary json file", type=str)
    parser.add_argument("--post", help="option for postprocessing", type=str)

    args =  parser.parse_args()
    job_id = args.job_id
    batch = args.batch
    port = args.port
    path_json = args.path_json
    post = args.post

    try:
        abs_path=os.path.join(path_json,'temp.json')
        with open(abs_path, 'r') as f:
            config = json.load(f, object_pairs_hook=OrderedDict)
    except:
        raise FileNotFoundError("temp.json did not load properly.")

    path_to_post_post_file = os.path.join(path_json,'post_post/post_post.py')

## Run the simulations via object methods
    driver_obj = Driver.from_config_create_driver(config, job_id, batch, port, path_to_post_post_file)

    driver_obj.main_run()
    if post == 'true':
        driver_obj.finish_and_clean()

######## Helper functions ##############################

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
