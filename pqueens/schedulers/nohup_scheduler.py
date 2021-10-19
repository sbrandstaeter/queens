import os
import sys

from pqueens.utils.run_subprocess import run_subprocess

from .scheduler import Scheduler


class NohupScheduler(Scheduler):
    """
    Scheduler for nohup QUEENS jobs

    Args:
        base_settings (dict): Configurations that are set in the base class and are partly used
                              in this class
        scheduler_name (str): Name of the scheduler as specified in input file

    Attributes:
        name (str): Name of the scheduler as specified in input file
    """

    def __init__(self, base_settings, scheduler_name):
        self.name = scheduler_name
        super(NohupScheduler, self).__init__(base_settings)

    @classmethod
    def from_config_create_scheduler(
        cls, config, base_settings, scheduler_name=None
    ):  # TODO scheduler name
        # is depreciated
        """ Create scheduler from config dictionary

        Args:
            scheduler_name (str):   (Optional) Name of scheduler
            config (dict):          Dictionary containing problem description of input file
            base_settings (dict): Configurations that are set in the base class and are partly
                                  reused to construct this class

        Returns:
            scheduler_obj (obj): Instance of LocalScheduler

        """

        # pre assemble some strings as base_settings
        base_settings['scheduler_start'] = None
        base_settings['scheduler_options'] = None

        return cls(base_settings, scheduler_name)
