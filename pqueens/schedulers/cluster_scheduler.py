
import pqueens
import subprocess
import sys
import json
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
    def submit_command(self,scheduler_options):
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
               driver_options, database_address):
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
            driver_options (dict):
                Options for driver
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

        # note for posterity getting a dictionary with options properly from here
        # to the driver script is a pain a geetting the quotations right is a
        # nightmare. In any case the stuff below works, so do not touch it

        # convert driver options dict to json
        driver_options_json_str = json.dumps(driver_options)
        # run it a second time (not quite sure why this is needed, but it
        # does not work without it)
        driver_options_json_str = json.dumps(driver_options_json_str)
        driver_options_json_str = "\\'" +driver_options_json_str  + "\\'"

        # the '<' is needed for execution of local python scripts on potentially remote servers
        driver_args = '-F ' + driver_options_json_str
        # one more time
        driver_args = json.dumps(driver_args)
        run_command = ['<', "/Users/jonas/work/adco/queens_code/pqueens/pqueens/drivers/dummy_driver_baci_pbs_kaiser.py" ,driver_args]

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
