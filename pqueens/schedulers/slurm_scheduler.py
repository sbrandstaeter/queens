""" Here should be a docstring """

import sys
import os
from .scheduler import Scheduler
from pqueens.utils.run_subprocess import run_subprocess


class SlurmScheduler(Scheduler):
    """
    Interface to SLURM based queuing systems to submit and query jobs.

    This class provides a basic interface to the Slurm job queuing system to submit
    and query jobs to a cluster. This also works if the cluster is a remote
    resource that has to be connected to via ssh. When submitting the job, the
    process id is returned to enable queries about the job status later on.

    Returns:
        SlurmScheduler (obj): Instance of the SlurmScheduler Class

    """

    def __init__(self, scheduler_name, base_settings):
        super(SlurmScheduler, self).__init__(base_settings)

    @classmethod
    def from_config_create_scheduler(cls, config, base_settings, scheduler_name=None):
        """ Create Slurm-scheduler from problem dictionary

        Args:
            scheduler_name (str):   name of scheduler
            config (dict):          dictionary containing problem description
            base_settings (dict): Dictionary containing some basic setting for parent class
                                  (depreciated: will be changed soon)

        Returns:
            scheduler (obj):              instance of SlurmScheduler

        """
        scheduler_options = base_settings['options']
        # read necessary variables from config
        num_procs = scheduler_options['num_procs']
        walltime = scheduler_options['walltime']
        cluster_script = scheduler_options['cluster_script']
        if (
            scheduler_options['scheduler_output'].lower() == 'true'
            or scheduler_options['scheduler_output'] == ""
        ):
            output = ""
        elif scheduler_options['scheduler_output'].lower() == 'false':
            output = '--output=/dev/null --error=/dev/null'
        else:
            raise RuntimeError(
                r"The Scheduler requires a 'True' or 'False' value for the slurm_output parameter"
            )

        # pre assemble some strings as base_settings
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        rel_path = '../utils/jobscript_slurm.sh'
        abs_path = os.path.join(script_dir, rel_path)

        base_settings['scheduler_template'] = abs_path
        base_settings['scheduler_start'] = 'sbatch'
        base_settings['scheduler_options'] = {}
        base_settings['scheduler_options']['output'] = output
        base_settings['scheduler_options']['ntasks'] = num_procs
        base_settings['scheduler_options']['walltime'] = walltime
        base_settings['scheduler_options']['job_name'] = None  # real name will be assembled later
        base_settings['scheduler_options']['CLUSTERSCRIPT'] = cluster_script

        return cls(scheduler_name, base_settings)

    def get_cluster_job_id(self, output):
        """
        Helper function to retrieve job_id information after
        submitting a job to the job scheduling software

        Args:
            output (string): Output returned when submitting the job

        Returns:
            match object (str): with regular expression matching job id

        """
        regex = output.split()
        return regex[-1]

    def alive(self, process_id):  # TODO method might me depreciated!
        """ Check whether job is alive
        The function checks if job is alive. If it is not i.e., the job is
        either on hold or suspended the function will attempt to kill it

        Args:
            process_id (int): id of process associated with job

        Returns:
            bool: is job alive or dead

        """

        alive = False
        try:
            # join lists
            command_list = [self.connect_to_resource, 'squeue --job', str(process_id)]
            command_string = ' '.join(command_list)
            _, p, stdout, _ = run_subprocess(command_string)
            output2 = stdout.split()
            # second to last entry is (should be )the job status
            status = output2[-4]  # TODO: Check if that still holds
        except ValueError:
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
                _, p, stdout, stderr = run_subprocess(command_string)
                print(stdout)
                sys.stderr.write("Killed job %d.\n" % (process_id))
            except ValueError:
                sys.stderr.write("Failed to kill job %d.\n" % (process_id))

            return False
        else:
            return True
