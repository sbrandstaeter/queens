

import argparse
import os
import time
import sys
try: import simplejson as json
except ImportError: import json

from resources.resource import parse_resources_from_configuration
from resources.resource import print_resources_status

from database.mongodb import MongoDB

from collections import OrderedDict

def get_options():
    parser = argparse.ArgumentParser(description="QUEENS")
    parser.add_argument('--input', type=str, default='input.json',
                        help='Input file in .json format.')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory to write resutls to.')
    parser.add_argument('--debug', type=str, default='no',
                        help='debug mode yes/no')

    args = parser.parse_args()

    input_file  = os.path.realpath(os.path.expanduser(args.input))
    try:
        with open(input_file, 'r') as f:
            options = json.load(f, object_pairs_hook=OrderedDict)
    except:
        raise Exception("config.json did not load properly.")

    if args.output_dir is None:
        raise Exception("No output directory was given.")

    output_dir  = os.path.realpath(os.path.expanduser(args.output_dir))
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
    options = get_options()
    #print("options {}".format(options))

    # create resource
    resources = parse_resources_from_configuration(options)

     # connect to the database
    db_address = options['database']['address']
    experiment_name = options['experiment-name']

    sys.stderr.write('Using database at %s.\n' % db_address)
    db   = MongoDB(database_address=db_address)

    # create dummy job
    new_job = get_dummy_suggestion(db,experiment_name,options,'my-machine')

    # Submit the job to the appropriate resource

    process_id = resources['my-machine'].attempt_dispatch(experiment_name, new_job, db_address, '/Users/jonas/work/adco/queens_code/pqueens/example_simulator_functions')
    print(options['output_dir'])

    # Set the status of the job appropriately (successfully submitted or not)
    if process_id is None:
        new_job['status'] = 'broken'
        save_job(new_job, db, experiment_name)
    else:
        new_job['status'] = 'pending'
        new_job['proc_id'] = process_id
        save_job(new_job, db, experiment_name)

    jobs = load_jobs(db, experiment_name)


def get_dummy_suggestion(db,experiment_name,options,resource_name):

    jobs = load_jobs(db, experiment_name)

    job_id = len(jobs) + 1

    job = {
        'id'          : job_id,
        'params'      : {'x':{ "type" : "FLOAT",
                               "size" : 1,
                               "values" : 2.5},
                         'y':{ "type" : "FLOAT",
                                        "size" : 1,
                                        "values" : 1.0}},
        'expt_dir'    : '/Users/jonas/work/adco/queens_code/pqueens/example_simulator_functions',
        #'tasks'       : task_names,
        'resource'    : resource_name,
        'main-file'   : options['main-file'],
        'language'    : options['language'],
        'status'      : 'new',
        'submit time' : time.time(),
        'start time'  : None,
        'end time'    : None
    }

    save_job(job, db, experiment_name)

    return job

def load_jobs(db, experiment_name):
    """load the jobs from the database

    Returns
    -------
    jobs : list
        a list of jobs or an empty list
    """
    jobs = db.load(experiment_name, 'jobs')

    if jobs is None:
        jobs = []
    if isinstance(jobs, dict):
        jobs = [jobs]

    return jobs

def save_job(job, db, experiment_name):
    """save a job to the database"""
    db.save(job, experiment_name, 'jobs', {'id' : job['id']})




if __name__ == '__main__':
    main()
