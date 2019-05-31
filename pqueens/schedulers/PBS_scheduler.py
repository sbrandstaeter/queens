import sys
import subprocess
import re
from .schedulers.scheduler import Scheduler

class PBSScheduler(Scheduler):
    """ Minimal interface to Torque queing system to submit and query jobs

    This class provides a basic interface to the PBS job queing system to submit
    and query jobs to a cluster. This also works if the cluster is a remote
    resource that has to be connected to via ssh. When submitting the job, the
    process id is returned to enable queries about the job status later on.

    as of now this scheduler is written specifically for the LNM Kaiser cluster,
    but can serve as an example for other Torque queueing systems

    Attributes:
        connect_to_resource (list): list containing commands to
                                    connect to resource
    """

    def __init__(self, scheduler_name, base_settings):
        """
        Args:
            scheduler_name (string):    Name of Scheduler
            num_procs_per_node (int):   Number of procs per node
            num_nodes (int):            Number of nodes
            walltime (string):          Wall time in hours
            user_mail (string):         Email adress of user
            queue (string):             Name of queue
            connect_to_resource (list): list containing commands to
                                        connect to resaurce
        """
        super(PBSScheduler, self).__init__(base_settings)
        self.name = scheduler_name
        self.connect_to_resource = connect_to_resource

    @classmethod
    def from_config_create_scheduler(cls, config, base_setings,scheduler_name=None):
        """ Create PBS scheduler from config dictionary

        Args:
            scheduler_name (str):   name of scheduler
            config (dict):          dictionary containing problem description

        Returns:
            scheduler:              instance of PBSScheduler
        """
        options = config[scheduler_name]
        connect_to_resource = options["connect_to_resource"]
        return cls(scheduler_name, connect_to_resource, base_settings)

########### auxiliary methods #################################################

    def output_regexp(self):
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
        regex = r'(^\d+)'
        return re.search(regex, output)

######## children methods that need to be implemented

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

    def alive(self, process_id): # TODO: This methods needs to be checked as might not be called properly
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
            command_list = self.connect_to_resource + ['qstat', str(process_id)]
            command_string = ' '.join(command_list)
            stdout, stderr, p = super run_subprocess(command_string)
            output2 = stdout.split()
            # second to last entry is (should be )the job status
            status = output2[-2]
        except:
            # job not found
            status = -1
            sys.stderr.write("EXC: %s\n" % str(sys.exc_info()[0]))
            sys.stderr.write("Could not find job for process id %d\n" % process_id)

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
                command_list = self.connect_to_resource + ['qdel', str(process_id)]
                command_string = ' '.join(command_list)
                stdout, stderr, p = super run_subprocess(command_string)
                print(stdout)
                sys.stderr.write("Killed job %d.\n" % (process_id))
            except:
                sys.stderr.write("Failed to kill job %d.\n" % (process_id))

            return False
        else:
            return True
