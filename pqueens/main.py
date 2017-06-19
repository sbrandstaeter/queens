
import pqueens
import argparse
import os
import time
import sys
import numpy as np
try: import simplejson as json
except ImportError: import json
from  pqueens.emulators.gp_emulator import GPEmulator
from  resources.resource import parse_resources_from_configuration
from  resources.resource import print_resources_status
from designers.designer_factory import DesignerFactory

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

    # create resource
    resources = parse_resources_from_configuration(options)

     # connect to the database
    db_address = options['database']['address']
    experiment_name = options['experiment-name']

    sys.stderr.write('Using database at %s.\n' % db_address)
    db   = MongoDB(database_address=db_address)

    # create designer
    designer = DesignerFactory.create_designer(options['designer']['type'],
                                               options['variables'],
                                               options['designer']['seed'],
                                               options['designer']['num_samples'])

    suggester = designer.sample_generator()

    jobs = load_jobs(db, experiment_name)

    for suggestion in suggester:
        processed_suggestion = False
        while not processed_suggestion:
            if resources['my-machine'].accepting_jobs(jobs):

                new_job = create_new_job(db,suggestion,experiment_name,options,'my-machine')

                # Submit the job to the appropriate resource
                process_id = resources['my-machine'].attempt_dispatch(experiment_name,
                                                                      new_job, db_address,
                                                                      options['output_dir'])

                # Set the status of the job appropriately (successfully submitted or not)
                if process_id is None:
                    new_job['status'] = 'broken'
                    save_job(new_job, db, experiment_name)
                else:
                    print("suggested_job[status{} ".format(new_job['status']))
                    new_job['status'] = 'pending'
                    new_job['proc_id'] = process_id
                    save_job(new_job, db, experiment_name)

                processed_suggestion = True
                jobs = load_jobs(db, experiment_name)
                print_resources_status(resources, jobs)

            else: # tired(db, experiment_name, resources['my-machine']):
                time.sleep(options.get('polling-time', 5))
                jobs = load_jobs(db, experiment_name)


    #print_resources_status(resources, jobs)
    time.sleep(options.get('polling-time', 30))
    build_emulator(db, experiment_name, options)
    # create emulator
    # loop over all jobs, get param values and corresponding solution data
    # fit gp_emulator
    # compute emulator based results


def get_dummy_suggestion(db,suggester,experiment_name,options,resource_name):

    jobs = load_jobs(db, experiment_name)

    job_id = len(jobs) + 1

    variable_values = next(suggester)
    i = 0
    params = {}
    for key, variable_meta_data in options['variables'].items():
        params[key] = variable_meta_data
        params[key]['values']=variable_values[i]
        i+=1

    job = {
        'id'           : job_id,
        'params' :       params,
        'expt_dir'     : options['output_dir'],
        'expt_name'    : experiment_name,
        'resource'     : resource_name,
        'driver_type'  : options['driver']['driver_type'],
        'driver_params': options['driver']['driver_params'],
        'status'       : 'new',
        'submit time'  : time.time(),
        'start time'   : None,
        'end time'     : None
    }

    save_job(job, db, experiment_name)

    return job

def create_new_job(db,suggestion,experiment_name,options,resource_name):

    print("Created new job")
    jobs = load_jobs(db, experiment_name)

    job_id = len(jobs) + 1

    i = 0
    params = {}
    for key, variable_meta_data in options['variables'].items():
        params[key] = variable_meta_data
        params[key]['values']=suggestion[i]
        i+=1

    job = {
        'id'           : job_id,
        'params' :       params,
        'expt_dir'     : options['output_dir'],
        'expt_name'    : experiment_name,
        'resource'     : resource_name,
        'driver_type'  : options['driver']['driver_type'],
        'driver_params': options['driver']['driver_params'],
        'status'       : 'new',
        'submit time'  : time.time(),
        'start time'   : None,
        'end time'     : None
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

def tired(db, experiment_name, resource):
    """
    return True if resources is not accepting jobs
    """
    jobs = load_jobs(db, experiment_name)
    if resource.accepting_jobs(jobs):
        return False
    return True

def build_emulator(db, experiment_name, options):
    jobs = load_jobs(db, experiment_name)
    num_jobs = len(jobs)
    num_param = len(options['variables'])
    # init arrays
    y = np.zeros((num_jobs,1))
    x = np.zeros((num_jobs,num_param))
    i = 0
    for job in jobs:
        y[i] = job['result']
        j=0
        print("job['params']{} ".format(job['params']))
        for param in job['params']:
            print("param{} ".format(param))
            x[i,j]=job['params'][param]['values']
            j+=1
            #params[key]['values']=suggestion[i]

        i+=1
    my_emulator = GPEmulator(x,y,options['variables'])
    mean_mean, var_mean = my_emulator.compute_mean()
    print('Mean{} '.format(mean_mean))
    print('Variance of Mean{} '.format(var_mean))

    return my_emulator






if __name__ == '__main__':
    main()
