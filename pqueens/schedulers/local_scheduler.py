""" There should be a docstring """

import os
from .scheduler import Scheduler


class LocalScheduler(Scheduler):
    """ Scheduler which submits jobs to the local machine via a shell command"""

    def __init__(self, base_settings, scheduler_name):
        self.name = scheduler_name
        super(LocalScheduler, self).__init__(base_settings)

    @classmethod
    def from_config_create_scheduler(cls, config, base_settings, scheduler_name=None):
        """ Create scheduler from config dictionary

        Args:
            scheduler_name (str):   Name of scheduler
            config (dict):          Dictionary containing problem description

        Returns:
            scheduler:              Instance of LocalScheduler
        """
        # pre assemble some strings as base_settings
        base_settings['output'] = None
        base_settings['tasks_info'] = None
        base_settings['walltime_info'] = None
        base_settings['job_flag'] = None
        base_settings['scheduler_start'] = None
        base_settings['command_line_opt'] = None
        base_settings['cluster_bind'] = None

        return cls(base_settings, scheduler_name)

# ------------------- CHILD METHODS THAT MUST BE IMPLEMENTED ------------------
    def alive(self, process_id):  # TODO: ok for now (gets called in resources)
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

    def get_process_id_from_output(self, output):
        """ docstring """
        pass
