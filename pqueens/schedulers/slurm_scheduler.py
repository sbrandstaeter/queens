import sys
import subprocess
import re
from pqueens.schedulers.cluster_scheduler import AbstractClusterScheduler


class SlurmScheduler(AbstractClusterScheduler):
    """ Minimal interface to SLURM queing system to submit and query jobs

    This class provides a basic interface to the Slurm job queing system to submit
    and query jobs to a cluster. This also works if the cluster is a remote
    resource that has to be connected to via ssh. When submitting the job, the
    process id is returned to enable queries about the job status later on.

    This scheduler is written specifically for the LNM Bruteforce cluster

    Attributes:
        connect_to_resource (list): list containing commands to
                                    connect to resource
    """

    def __init__(self, scheduler_name, num_procs_per_node, num_nodes, walltime, user_mail, connect_to_resource, output):
        """
        Args:
            scheduler_name (string):    Name of Scheduler
            num_procs_per_node (int):   Number of procs per node
            num_nodes (int):            Number of nodes
            walltime (string):          Wall time in hours
            user_mail (string):         Email adress of user
            connect_to_resource (list): list containing commands to
                                        connect to resource
            output (boolean):           Flag for slurm output
        """
        super(SlurmScheduler, self).__init__()
        self.name = scheduler_name
        self.num_procs_per_node = num_procs_per_node
        self.num_nodes = num_nodes
        self.walltime = walltime
        self.user_mail = user_mail
        self.connect_to_resource = connect_to_resource
        self.output = output

    @classmethod
    def from_config_create_scheduler(cls, scheduler_name, config):
        """ Create Slurm scheduler from config dictionary

        Args:
            scheduler_name (str):   name of scheduler
            config (dict):          dictionary containing problem description

        Returns:
            scheduler:              instance of SlurmScheduler
        """
        options = config[scheduler_name]

        num_procs_per_node = options['num_procs_per_node']
        num_nodes = options['num_nodes']
        walltime = options['walltime']
        user_mail = options['email']
        connect_to_resource = options["connect_to_resource"]
        output = options["slurm_output"]

        return cls(scheduler_name, num_procs_per_node, num_nodes, walltime, user_mail, connect_to_resource, output)

    def output_regexp(self): # TODO Check what this does exactly
        return r'(^\d+)'

    def get_process_id_from_output(self, output): # TODO: Check what this does
                                                  # exaclty
        """ Helper function to retrieve process id

            Helper function to retrieve after submitting a job to the job
            scheduling software
        Args:
            output (string): Output returned when submitting the job

        Returns:
            match object: with regular expression matching process id
        """
        #regex = r'(^\d+)'
        regex=output.split()
        return regex[-1]

    def submit_command(self, job_name):
        """ Get submit command for Slurm type scheduler

            The function actually prepends the commands necessary to connect to
            the resource to enable remote job submission
        Args:
            job_name (string): name of job to submit

        Returns:
            list: Submission command(s)
        """
        # TODO: CHECK dev\null
        # pre assemble some strings
        proc_info = '--nodes={} --ntasks={}'.format(self.num_nodes, self.num_procs_per_node)
        walltime_info = '--time={}'.format(self.walltime)
        mail_info = '--mail-user={}'.format(self.user_mail)
        job_info = '--job-name={}'.format(job_name)


        if self.output.lower()=="true" or self.output=="":
            command_list = self.connect_to_resource  \
                          + [r'sbatch --mail-type=ALL', mail_info, job_info, proc_info, walltime_info]
        elif self.output.lower()=="false":
            command_list = self.connect_to_resource  \
                          + [r'sbatch --mail-type=ALL --output=\dev\null --error=\dev\null', mail_info, job_info, proc_info, walltime_info]
        else:
            raise RuntimeError(r"The Scheduler requires a 'True' or 'False' value for the slurm_output parameter")

        return command_list

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
            command_list = self.connect_to_resource + ['squeue --job', str(process_id)]
            command_string = ' '.join(command_list)

            process = subprocess.Popen(command_string,
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       shell=True,
                                       universal_newlines=True)

            output, std_err = process.communicate()
            process.stdin.close()
            output2 = output.split()
            # second to last entry is (should be )the job status
            status = output2[-4] #TODO: Check if that still holds
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
                command_list = self.connect_to_resource + ['scancel', str(process_id)]
                command_string = ' '.join(command_list)
                process = subprocess.Popen(command_string,
                                           stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT,
                                           shell=True,
                                           universal_newlines=True)

                output, std_err = process.communicate()
                process.stdin.close()
                print(output)
                sys.stderr.write("Killed job %d.\n" % (process_id))
            except:
                sys.stderr.write("Failed to kill job %d.\n" % (process_id))

            return False
        else:
            return True
