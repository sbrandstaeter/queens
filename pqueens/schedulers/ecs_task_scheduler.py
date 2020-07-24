import os
import sys
from .scheduler import Scheduler


class ECSTaskScheduler(Scheduler):
    """
    Scheduler for QUEENS tasks on AWS ECS.

    Args:
        base_settings (dict): Configurations that are set in the base class and are partly used
                              in this class
        scheduler_name (str): Name of the scheduler as specified in input file

    Attributes:
        name (str): Name of the scheduler as specified in input file
    """

    def __init__(self, base_settings, scheduler_name):
        self.name = scheduler_name
        super(ECSTaskScheduler, self).__init__(base_settings)

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
        base_settings['output'] = None
        base_settings['tasks_info'] = None
        base_settings['walltime_info'] = None
        base_settings['job_flag'] = None
        base_settings['scheduler_start'] = None
        base_settings['command_line_opt'] = None
        base_settings['scheduler_options'] = {}
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        rel_path = '../utils/regtask_aws_docker_queens.json'
        abs_path = os.path.join(script_dir, rel_path)
        base_settings['scheduler_template'] = abs_path

        return cls(base_settings, scheduler_name)

    # ------------------- CHILD METHODS THAT MUST BE IMPLEMENTED ------------------
    def alive(self, process_id):  # TODO: seems not to be used
        """
        Not necessary for AWS ECS scheduler but mandatory for parent class initialization

        Returns:
            None

        """
        pass

    def get_process_id_from_output(self):
        """
        Not necessary for AWS ECS scheduler but mandatory for parent class initialization

        Returns:
            None

        """
        pass
