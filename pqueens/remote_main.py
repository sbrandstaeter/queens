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
    args =  parser.parse_args()
    job_id = args.job_id
    batch = args.batch

## Check if all necessary temp files are available
# --> compare with paths in json file
# --> copy json file in the same directory as this remote_main file so that we can check via reltive path
    try:
        with open('temp.json', 'r') as f:
            config = json.load(f, object_pairs_hook=OrderedDict)
    except:
        raise FileNotFoundError("temp.json did not load properly.")

## Create Driver and Scheduler object from input JSON temp file
#TODO: Somehow a database object needs to be passed to driver --> Check how this is done in scheduler scheduler = Scheduler.from_config_create_scheduler(options) # creates driver_obj as well
## Run the simulations via object methods
    driver_obj = Driver.from_config_create_driver(config, job_id, batch) #TODO: Check if there is a better way than creating the object everytime 
    driver_obj.main_run(job_id,batch) #TODO what about slurm cmd?
## Clean-up and delete all temp files
    os.remove('temp.json')

######## Helper functions ##############################

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
