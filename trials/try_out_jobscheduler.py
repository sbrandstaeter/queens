from pqueens.schedulers.PBS_scheduler import PBSScheduler

connect_to_resource = ['ssh', '-T','-p 9001', 'biehler@localhost']
my_scheduler = PBSScheduler(connect_to_resource)

scheduler_options = {}
scheduler_options['num_procs_per_node'] = '16'
scheduler_options['num_nodes'] = '1'
scheduler_options['walltime'] = '300:00:00'
scheduler_options['email']     = 'biehler@lnm.mw.tum.de'
scheduler_options['queue']   = 'opteron'

driver_options = {}
driver_options['bla'] = '/home/biehler/baci/release/baci-release'

#PBS -M biehler@lnm.mw.tum.de
#PBS -m abe
#PBS -N queens_run_1
#PBS -l nodes=1:ppn=16
#PBS -l walltime=300:00:00
#PBS -q opteron

my_scheduler.submit(1,'test','test2',scheduler_options,
                    driver_options,'database_adress')
#my_scheduler.alive(322932)
