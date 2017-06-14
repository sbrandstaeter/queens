from pqueens.schedulers.PBS_scheduler import PBSScheduler

connect_to_resource = ['ssh', '-T','-p 9001', 'biehler@localhost']
my_scheduler = PBSScheduler(connect_to_resource)
my_scheduler.submit(1,'test','test2','database_adress')
#my_scheduler.alive(322932)
