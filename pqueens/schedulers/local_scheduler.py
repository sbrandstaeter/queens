
#import pqueens.resources.resource
from .abstract_scheduler import AbstractScheduler

import os
import subprocess
import sys
import pathlib


class LocalScheduler(AbstractScheduler):
    """ Scheduler which submits jobs to the local machine via a shell command"""


    def submit(self, job_id, experiment_name, experiment_dir, database_address):
        """ Submit job locally by calling subprocess

        Args:
            job_id (int):               Id of job to be started
            experiment_name (string):   Name of experiment
            experiment_dir (string):    Directory to write output to
            database_address (string):  Address of MongoDB database

        Returns:
            int: id of process associated with the job,
                 or None if submission failed

        """
        # TODO find a better way to do this
        #base_path = os.path.dirname(os.path.realpath(pqueens.__file__))
        base_path = pathlib.Path(__file__).parent.parent

        # assemble shell command
        cmd = ('python %s/launcher.py --db_address %s --experiment_name '
               '%s --job_id %s' %
               (base_path, database_address, experiment_name, job_id))

        output_directory = os.path.join(experiment_dir, 'output')
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)

        output_filename = os.path.join(output_directory, '%08d.out' % job_id)
        output_file = open(output_filename, 'w')

        print("starting process")
        print("output_file{}".format(output_file))
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
