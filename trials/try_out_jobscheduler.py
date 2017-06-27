from pqueens.schedulers.PBS_scheduler import PBSScheduler

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
driver_options['bla'] = '/home/biehler/baci/release/baci-release'

# path_to_executable
driver_options['executable'] = '/home/biehler/baci/release/baci-release'

# path_to_postprocessor
driver_options['post_processor'] = '/home/biehler/baci/release/post_drt_monitor'

# path_to_input_file_template
driver_options['input_template'] = '/home/biehler/input/input2.dat'

# experiment_dir
driver_options['experiment_dir'] = '/home/biehler/queens_testing/my_first_queens_jobqueens'

# job_id
job_id = 1

# post_processing options
driver_options['post_process_command'] = 'stuff'


my_scheduler.submit(job_id,'queens_first_try','/home/biehler/queens_testing/my_first_queens_jobqueens_job_1/',scheduler_options,
                    driver_options,'database_adress')

#my_scheduler.alive(322932)
