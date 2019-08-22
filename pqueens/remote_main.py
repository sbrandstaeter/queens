# Imports
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
    args =  parser.parse_args()
    job_id = args.job_id
    batch = args.batch
    port = args.port
    path_json = args.path_json

## Check if all necessary temp files are available
# --> compare with paths in json file
# --> copy json file in the same directory as this remote_main file so that we can check via reltive path
    try:
        #dir_path = os.path.dirname(__file__)
        #abs_path = os.path.join(dir_path,'temp.json')
        abs_path=os.path.join(path_json,'temp.json')
        with open(abs_path, 'r') as f:
            config = json.load(f, object_pairs_hook=OrderedDict)
    except:
        raise FileNotFoundError("temp.json did not load properly.")

## Create Driver and Scheduler object from input JSON temp file
#TODO: Somehow a database object needs to be passed to driver --> Check how this is done in scheduler scheduler = Scheduler.from_config_create_scheduler(options) # creates driver_obj as well
## Run the simulations via object methods
    driver_obj = Driver.from_config_create_driver(config, job_id, batch, port) #TODO: Check if there is a better way than creating the object everytime
    driver_obj.main_run() #TODO what about slurm cmd?

######## Helper functions ##############################

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
