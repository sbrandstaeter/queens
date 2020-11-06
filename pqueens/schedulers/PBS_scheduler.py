import sys
import os
from .scheduler import Scheduler
from pqueens.utils.run_subprocess import run_subprocess


class PBSScheduler(Scheduler):
    """
    Scheduler interface for Torque/PBS-based queuing systems to submit and query jobs.

    This class provides a basic interface to the PBS job queuing system to submit
    and query jobs to a cluster. This also works if the cluster is a remote
    resource that has to be connected to via ssh. When submitting the job, the
    process id is returned to enable queries about the job status later on.

    Attributes:
        connect_to_resource (list): list containing commands to
                                    connect to resource
    """

    def __init__(self, scheduler_name, base_settings):
        super(PBSScheduler, self).__init__(base_settings)

    @classmethod
    def from_config_create_scheduler(cls, config, base_settings, scheduler_name=None):
        """ Create PBS scheduler from config dictionary

        Args:
            scheduler_name (str):   name of scheduler
            config (dict):          dictionary containing problem description

        Returns:
            scheduler:              instance of PBSScheduler

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
        rel_path = '../utils/jobscript_pbs.sh'
        abs_path = os.path.join(script_dir, rel_path)

        base_settings['scheduler_template'] = abs_path
        base_settings['scheduler_start'] = 'qsub'
        base_settings['scheduler_options'] = {}
        base_settings['scheduler_options']['output'] = output
        base_settings['scheduler_options']['ntasks'] = num_procs
        base_settings['scheduler_options']['walltime'] = walltime
        base_settings['scheduler_options']['job_name'] = None
        base_settings['scheduler_options']['CLUSTERSCRIPT'] = cluster_script

        return cls(scheduler_name, base_settings)

    # ----------------------------- AUXILIARY METHODS -----------------------------
    def get_cluster_job_id(self, output):
        """
        Helper function to retrieve job id from output string

            Helper function to retrieve after submitting a job to the job
            scheduling software

        Args:
            output (string): Output returned when submitting the job

        Returns:
            match object (str): with regular expression matching job id

        """
        return output.split('.')[0]

    # ---------------- CHILDREN METHODS THAT NEED TO BE IMPLEMENTED ---------------
    def alive(
        self, process_id
    ):  # TODO: This methods needs to be checked as might not be called properly
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
            command_list = self.connect_to_resource + ['qstat', str(process_id)]
            command_string = ' '.join(command_list)
            _, p, stdout, stderr = run_subprocess(command_string)
            output2 = stdout.split()
            # second to last entry is (should be )the job status
            status = output2[-2]
        except ValueError:
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
                _, p, stdout, stderr = run_subprocess(command_string)
                sys.stderr.write("Killed job %d.\n" % (process_id))
            except:
                sys.stderr.write("Failed to kill job %d.\n" % (process_id))

            return False
        else:
            return True
