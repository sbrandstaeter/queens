import os
import sys
import pathlib
from pqueens.schedulers.scheduler import Scheduler

class LocalScheduler(Scheduler):
    """ Scheduler which submits jobs to the local machine via a shell command"""

    def __init__(self, scheduler_name):
        """ Create LocalScheduler

        Args:
            scheduler_name (string):    Name of scheduler
            num_procs_per_node (int):   Number of procs per node
            walltime (string):          Wall time in hours
            output (boolean):           Flag for meta output
        """
        super(LocalScheduler, self).__init__() #TODO: Check if correct as wasnt here before
        self.name = scheduler_name
        self.num_procs_per_node = num_procs_per_node
        self.walltime = walltime
        self.output = output

    @classmethod
    def from_config_create_scheduler(cls, scheduler_name, config):
        """ Create scheduler from config dictionary

        Args:
            scheduler_name (str):   Name of scheduler
            config (dict):          Dictionary containing problem description

        Returns:
            scheduler:              Instance of LocalScheduler
        """
        options = config[scheduler_name]
        num_procs_per_node = options['num_procs_per_node']
        walltime = options['walltime']
        output = options["meta_output"]

        return cls(scheduler_name, num_procs_per_node, walltime, output)

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



    def submit(self, job_name):
        """ Get submit command for local scheduler

            The function actually prepends the commands necessary to
            start the driver with the actual simulation description
        Args:
            job_name (string): name of job to submit

        Returns:
            list: Submission command(s)
        """
        # TODO: check with instance writes the output if no slurm is used
        if self.output.lower()=="true" or self.output=="":
            command_list = [r'mpirun -np', self.num_procs_per_node]
        elif self.output.lower()=="false":
            command_list = [r'mpirun -np', self.num_procs_per_node] # TODO supress output
        else:
            raise RuntimeError(r"The Scheduler requires a 'True' or 'False' value for the slurm_output parameter")


        return command_list


    def alive(self, process_id):
        """ Check whether or not job is still running

        Args:
            process_id (int): id of process associated to job

        Returns:
            bool: indicator if job is still alive
        """
        try:
            # Send an alive signal to proc
            os.kill(process_id, 0) # TODO check if we find a better solution for that
        except OSError:
            return False
        else:
            return True
