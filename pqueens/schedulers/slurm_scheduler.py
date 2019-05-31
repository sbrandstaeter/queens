import sys
import subprocess
from .schedulers.scheduler import Scheduler

class SlurmScheduler(Scheduler):
    """ Minimal interface to SLURM queing system to submit and query jobs

    This class provides a basic interface to the Slurm job queing system to submit
    and query jobs to a cluster. This also works if the cluster is a remote
    resource that has to be connected to via ssh. When submitting the job, the
    process id is returned to enable queries about the job status later on.

    This scheduler is written specifically for the LNM Bruteforce cluster but can be used
    as an example for other Slurm based systems

    """

    def __init__(self, scheduler_name, connect_to_ressource, base_settings):
        """
        Args:
            scheduler_name (string):    Name of Scheduler
            num_procs_per_node (int):   Number of procs per node
            num_nodes (int):            Number of nodes
            walltime (string):          Wall time in hours
            user_mail (string):         Email adress of user
            output (boolean):           Flag for slurm output
        """
        super(SlurmScheduler, self).__init__(base_settings)
        # TODO: Does the latter trigger the abstact cluster scheduler??
        self.name = scheduler_name
        self.connect_to_ressource

    @classmethod
    def from_config_create_scheduler(cls, config, base_settings, scheduler_name=None):
        """ Create Slurm scheduler from config dictionary

        Args:
            scheduler_name (str):   name of scheduler
            config (dict):          dictionary containing problem description

        Returns:
            scheduler:              instance of SlurmScheduler
        """
        options = config[scheduler_name]
        connect_to_ressource = options['connect_to_ressource']
        return cls(scheduler_name, connect_to_ressource, base_settings)

###### auxiliary methods #################################################
    def output_regexp(self): # TODO Check what this does exactly
        return r'(^\d+)'

    def get_process_id_from_output(self, output):
        """ Helper function to retrieve process id

            Helper function to retrieve after submitting a job to the job
            scheduling software
        Args:
            output (string): Output returned when submitting the job

        Returns:
            match object: with regular expression matching process id
        """
        regex=output.split()
        return regex[-1]

########### Children methods that need to be implemented #######################
def alive(self, process_id):
        """ Check whether job is alive

        The function checks if job is alive. If it is not i.e., the job is
        either on hold or suspended the fuction will attempt to kill it

        Args:
            process_id (int): id of process associated with job

        Returns:
            bool: is job alive or dead
        """

        alive = False
        try:
            # join lists
            command_list = [self.connect_to_ressource,'squeue --job', str(process_id)]
            command_string = ' '.join(command_list)
            stdout, stderr, p = super run_subprocess(command_string)
            output2 = stdout.split()
            # second to last entry is (should be )the job status
            status = output2[-4] #TODO: Check if that still holds
            print('This is a test output')
        except:
            # job not found
            status = -1
            sys.stderr.write("EXC: %s\n" % str(sys.exc_info()[0]))
            sys.stderr.write("Could not find job for process id %d\n" % process_id)
            print('job wasnt found')

        if status == 'Q':
            sys.stderr.write("Job %d waiting in queue.\n" % (process_id))
            alive = True
        elif status == 'R':
            sys.stderr.write("Job %d is running.\n" % (process_id))
            alive = True
        elif status in ['H', 'S']:
            sys.stderr.write("Job %d is held or suspended.\n" % (process_id))
            alive = False

        if not alive:
            try:
                # try to kill the job.
                command_list = self.connect_to_resource + ['scancel', str(process_id)]
                command_string = ' '.join(command_list)
                stdout, stderr, p = super run_subprocess(command_string)
               print(stdout)
                sys.stderr.write("Killed job %d.\n" % (process_id))
            except:
                sys.stderr.write("Failed to kill job %d.\n" % (process_id))

            return False
        else:
            return True

    def submit(self, job_id, batch):
        """ Function to submit new job to scheduling software on a given resource


        Args:
            job_id (int):               Id of job to submit
            experiment_name (string):   Name of experiment
            batch (string):             Batch number of job
            experiment_dir (string):    Directory of experiment
            database_address (string):  Address of database to connect to
            driver_options (dict):      Options for driver

        Returns:
            int: proccess id of job

        """
        remote_args_list = '--job_id={} --batch={}'.format(job_id, batch) #TODO finalize args
        remote_args = ' '.join(remote_args_list)
        singularity = #TODO: Check how to switch to singularity env / container
        cmdlist_remote_main = [self.connect_to_ressource, singularity, './remote_main.py', remote_args]
        cmd_remote_main = ' '.join(cmdlist_remote_main)
        stdout, stderr, p = super run_subprocess(cmd_remote_main)

        # get the process id from text output
        match = self.get_process_id_from_output(stdout)
        try:
            return int(match)
        except:
            sys.stderr.write(output)
            return None
