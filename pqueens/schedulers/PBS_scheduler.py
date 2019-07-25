import sys
import subprocess
import re
from .scheduler import Scheduler

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
        self.connect_to_resource = base_settings["connect"]

    @classmethod
    def from_config_create_scheduler(cls, config, base_setings,scheduler_name=None):
        """ Create PBS scheduler from config dictionary

        Args:
            scheduler_name (str):   name of scheduler
            config (dict):          dictionary containing problem description

        Returns:
            scheduler:              instance of PBSScheduler
        """
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
            stdout, stderr, p = super().run_subprocess(command_string)
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
                stdout, stderr, p = super().run_subprocess(command_string)
                print(stdout)
                sys.stderr.write("Killed job %d.\n" % (process_id))
            except:
                sys.stderr.write("Failed to kill job %d.\n" % (process_id))

            return False
        else:
            return True
