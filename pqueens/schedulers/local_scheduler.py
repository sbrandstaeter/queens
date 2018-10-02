
import os
import subprocess
import sys
import pathlib

from pqueens.schedulers.scheduler import Scheduler

class LocalScheduler(Scheduler):
    """ Scheduler which submits jobs to the local machine via a shell command"""

    def __init__(self, scheduler_name):
        """ Create LocalScheduler

        Args:
            scheduler_name (string):    Name of scheduler
        """
        self.name = scheduler_name

    @classmethod
    def from_config_create_scheduler(cls, scheduler_name, config):
        """ Create scheduler from config dictionary

        Args:
            scheduler_name (str):   Name of scheduler
            config (dict):          Dictionary containing problem description

        Returns:
            scheduler:              Instance of LocalScheduler
        """

        return cls(scheduler_name)


    def submit(self, job_id, experiment_name, batch, experiment_dir,
               database_address, driver_option={}):
        """ Submit job locally by calling subprocess

        Args:
            job_id (int):               Id of job to be started
            experiment_name (string):   Name of experiment
            batch (string):             Batch number
            experiment_dir (string):    Directory to write output to
            database_address (string):  Address of MongoDB database
            driver_options (dict):      Options for driver (optional)

        Returns:
            int: id of process associated with the job,
                 or None if submission failed

        """
        # TODO implement proper driver hierarchy 
        # TODO find a better way to do this
        #base_path = os.path.dirname(os.path.realpath(pqueens.__file__))
        base_path = pathlib.Path(__file__).parent.parent

        # assemble shell command
        cmd = ('python %s/drivers/gen_driver_local.py --db_address %s --experiment_name '
               '%s --job_id %s --batch %s' %
               (base_path, database_address, experiment_name, job_id, batch))

        output_directory = os.path.join(experiment_dir, 'output')
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)

        output_filename = os.path.join(output_directory, '%08d.out' % job_id)
        output_file = open(output_filename, 'w')

        process = subprocess.Popen(cmd, stdout=output_file,
                                   stderr=output_file,
                                   shell=True)

        process.poll()
        if process.returncode is not None and process.returncode < 0:
            sys.stderr.write("Failed to submit job or job crashed "
                             "with return code %d !\n" % process.returncode)
            return None
        else:
            sys.stderr.write("Submitted job as process: %d\n" % process.pid)
            return process.pid


    def alive(self, process_id):
        """ Check whether or not job is still running

        Args:
            process_id (int): id of process associated to job

        Returns:
            bool: indicator if job is still alive
        """
        try:
            # Send an alive signal to proc
            os.kill(process_id, 0)
        except OSError:
            return False
        else:
            return True
