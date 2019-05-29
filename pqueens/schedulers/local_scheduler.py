
import os
import subprocess
import sys
import pathlib

from .schedulers.scheduler import Scheduler

class LocalScheduler(Scheduler):
    """ Scheduler which submits jobs to the local machine via a shell command"""

    def __init__(self, base_settings):
        super(LocalScheduler, self).__init__(base_settings)

    @classmethod
    def from_config_create_scheduler(cls, config, base_settings):
        """ Create scheduler from config dictionary

        Args:
            scheduler_name (str):   Name of scheduler
            config (dict):          Dictionary containing problem description

        Returns:
            scheduler:              Instance of LocalScheduler
        """

        return cls(base_settings)


######### abstract-methods that must be implemented #######################
    def alive(self, process_id): # TODO: ok for now (gets called in resources)
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

    def submit(self, job_id, experiment_name, batch, experiment_dir,
               database_address, driver_params={}):
        """ Submit job by calling corresponding Driver method
        """
        self.driver.main_run() # This is the only mehtod necessary: rest will be taken care of in the driver
