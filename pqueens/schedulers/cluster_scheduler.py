
import pqueens
import subprocess
import sys
#import socket
#import re
#import os

from abc import ABCMeta, abstractmethod


class AbstractClusterScheduler(object):
    """ Abstract base class for an interface to cluster scheduling software

    Attributes:
        connect_to_resource (list):     shell commands to connect resource
                                        running the scheduling software
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        """
        Args:
            connect_to_resource (list):     shell commands to connect resource
                                            running the scheduling software
        """
        self.connect_to_resource = []

    @abstractmethod
    def submit_command(self):
        pass

    @abstractmethod
    def output_regexp(self):
        pass

    @abstractmethod
    def get_process_id_from_output(self,output):
        pass

    @abstractmethod
    def alive(self, process_id):
        pass

    def submit(self, job_id, experiment_name, experiment_dir, scheduler_options,
               database_address):
        """ Function to submit new job to scheduling software on a given resource

        Args:
            job_id (int):
                id of job to submit
            experiment_name (string):
                name of experiment
            experiment_dir  (string):
                directory of experiment
            scheduler_options (dict):
                Options for scheduler
            database_address (string):
                address of database to connect to
        Returns (int): proccess id of job

        """
        #base_path = os.path.dirname(os.path.realpath(queens.__file__))
        # TODO add arguments to launcher such as experiment_name,
        # experiment_dir, database_address, and jobid

        # Since "localhost" might mean something different on the machine
        # we are submitting to, set it to the actual name of the parent machine
        #if database_address == "localhost":
        #   database_address = socket.gethostname()

        # the '<' is needed for execution of local python scripts on potentially remote servers
        run_command = ['<', '/Users/jonas/work/adco/queens_code/pqueens/pqueens/drivers/dummy_driver_baci_pbs_kaiser.py']

        # assemble job_name for cluster
        scheduler_options['job_name']='queens_{}_{}'.format(experiment_name,job_id)
        submit_command = self.submit_command(scheduler_options)

        submit_command.extend(run_command)

        command_string = ' '.join(submit_command)
        process = subprocess.Popen(command_string,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                shell=True,
                                universal_newlines = True)

        output, std_err = process.communicate()
        process.stdin.close()

        # get the process id from text output
        match = self.get_process_id_from_output(output)
        try:
            return int(match.group(1))
        except:
            sys.stderr.write(output)
            return None
