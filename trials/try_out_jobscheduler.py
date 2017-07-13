from pqueens.schedulers.PBS_scheduler import PBSScheduler
from pqueens.database.mongodb import MongoDB
import time
import numpy as np



variables =   {   "youngs" : {
                  "type" : "FLOAT",
                  "size" : 1,
                  "min"  : 0.1,
                  "max"  : 1.,
                  "distribution" : "normal",
                  "distribution_parameter" : [400000,10000]
                    },
                "beta" : {
                  "type" : "FLOAT",
                  "size" : 1,
                  "min"  : 1.0,
                  "max"  : 20,
                  "distribution" : "normal",
                  "distribution_parameter" : [400000,10000]
                    }
           }

connect_to_resource = ['ssh', '-T','-p 9001', 'biehler@localhost']
my_scheduler = PBSScheduler(connect_to_resource)

scheduler_options = {}
scheduler_options['num_procs_per_node'] = '16'
scheduler_options['num_nodes'] = '1'
scheduler_options['walltime'] = '300:00:00'
scheduler_options['email'] = 'biehler@lnm.mw.tum.de'
scheduler_options['queue'] = 'opteron'
scheduler_options['driver'] = '/Users/jonas/work/adco/queens_code/pqueens/pqueens/drivers/dummy_driver_baci_pbs_kaiser.py'

driver_options = {}
# path_to_executable
driver_options['executable'] = '/home/biehler/baci/release/baci-release'
# path_to_postprocessor
driver_options['post_processor'] = '/home/biehler/baci/release/post_drt_monitor'
# path_to_input_file_template
driver_options['input_template'] = '/home/biehler/input/input_template.dat'
# experiment_dir
driver_options['experiment_dir'] = '/home/biehler/queens_testing/my_first_test'
# post_processing options
driver_options['post_process_command'] = '--field=structure --node=3292 --start=1'

# set some parameters
experiment_name = 'queens_first_try'
experiment_dir = '/home/biehler/queens_testing/my_first_test'
# job_id
job_id = 1

# cauchy
#db   = MongoDB(database_address="129.187.58.39:27017")
# connect to port 27017 on cauchy via previously setup ssh tunnel
# to create ssh tunnel execute the following command on localhost
# (only if we are not in the lnm network)
# ssh -fN -L 27016:cauchy.lnm.mw.tum.de:27017 biehler@cauchy.lnm.mw.tum.de
db   = MongoDB(database_address="localhost:27016")

# create params dictionary
variable_values = np.array([0.3544,4.61])

i = 0
params = {}
for key, variable_meta_data in variables.items():
    params[key] = variable_meta_data
    params[key]['values']=variable_values[i]
    i+=1

job = {
    'id'           : job_id,
    'params'       : params,
    'expt_dir'     : driver_options['experiment_dir'] ,
    'expt_name'    : experiment_name,
    'resource'     : 'kaiser',
    'driver_type'  : 'baci_driver',
    'driver_options': driver_options,
    'status'       : 'new',
    'submit time'  : time.time(),
    'start time'   : None,
    'end time'     : None
}

db.save(job, experiment_name, 'jobs', {'id' : job['id']})

my_scheduler.submit(job_id,experiment_name,experiment_dir,scheduler_options,
                    driver_options,"129.187.58.39:27017")


test = db.load(experiment_name, 'jobs', {'id' : job['id']})

# TODO this does not yet work fix this 
while job['status'] is not 'complete':
    job = db.load(experiment_name, 'jobs', {'id' : job['id']})
    print('Job Information: {}'.format(job))
    time.sleep(20)
