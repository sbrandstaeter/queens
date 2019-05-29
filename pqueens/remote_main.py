# Imports
import argparse
import os
from pqueens.derivers.driver import Driver
try:
    import simplejson as json
except ImportError:
    import json

def main(args):
## Parse options from ssh command
   args = args.replace('\\', '\"') # TODO this is not nice but works (find better solution in the futer)
   args = json.loads(args) # Change json format to a dictionary format
   job_id = args['job_id']
   batch = args['batch']

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
    driver_obj = Driver.from_config_create_driver(config, job_id, batch)
    driver_obj.main_run(job_id,batch) #TODO what about slurm cmd?
## Clean-up and delete all temp files
    os.remove('temp.json')

######## Helper functions ##############################

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
