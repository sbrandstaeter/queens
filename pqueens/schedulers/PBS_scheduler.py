
import pqueens
import sys
import subprocess
import re
from .cluster_scheduler import AbstractClusterScheduler



class PBSScheduler(AbstractClusterScheduler):
    """ Minimal interface to Torque queing system to submit and query jobs

    This class provides a basic interface to the PBS job queing system to submit
    and query jobs to a cluster. This also works if the cluster is a remote
    resource that has to be connected to via ssh. When submitting the job, the
    process id is returned to enable queries about the job status later on.

    as of now this scheduler is written specifically for the LNM Kaiser cluster,
    but can serve as an example for other Torque queueing systems

    Attributes:
        connect_to_resource (list): list containing commands to
                                    connect to resaurce
    """

    def __init__(self, connect_to_resource_command=[]):
        """
        Args:
            connect_to_resource (list): list containing commands to
                                        connect to resaurce
        """
        super(PBSScheduler, self).__init__()
        self.connect_to_resource = connect_to_resource_command

    def output_regexp(self):
        return r'(^\d+)'

    def get_process_id_from_output(self,output):
        """ Helper function to retrieve process id

            Helper function to retrieve after submitting a job to the job
            scheduling software
        Args:
            output (string): Output returned when submitting the job

        Returns:
            match object: with regular expression matching process id
        """
        regex= r'(^\d+)'
        return re.search(regex, output)

    def submit_command(self):
        """ Get submit command for PBS type scheduler

            The function actually prepends the commands necessary to connect to
            the resource to enable remote job submission

        Returns:
            list: Submission command(s)
        """
        command_list = self.connect_to_resource + ['qsub']
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
            command_list = self.connect_to_resource + ['qstat', str(process_id)]
            command_string = ' '.join(command_list)

            process = subprocess.Popen(command_string,
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    shell=True,
                                    universal_newlines = True)

            output, std_err = process.communicate()
            process.stdin.close()
            output2 = output.split()
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
        elif status in ['H','S']:
            sys.stderr.write("Job %d is held or suspended.\n" % (process_id))
            alive = False

        if not alive:
            try:
                # try to kill the job.
                command_list = self.connect_to_resource + ['qdel', str(process_id)]
                command_string = ' '.join(command_list)
                process = subprocess.Popen(command_string,
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        shell=True,
                                        universal_newlines = True)

                output, std_err = process.communicate()
                process.stdin.close()
                print(output)
                sys.stderr.write("Killed job %d.\n" % (process_id))
            except:
                sys.stderr.write("Failed to kill job %d.\n" % (process_id))

            return False
        else:
            return True
