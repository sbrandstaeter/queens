import os
import sys

from pqueens.utils.injector import inject
from pqueens.utils.run_subprocess import run_subprocess
from pqueens.utils.script_generator import generate_submission_script
from pqueens.utils.string_extractor_and_checker import extract_string_from_output

from .scheduler import Scheduler


class ECSTaskScheduler(Scheduler):
    """
    AWS ECS task scheduler for QUEENS.

    """

    def __init__(self, base_settings):
        super(ECSTaskScheduler, self).__init__(base_settings)

    @classmethod
    def create_scheduler_class(cls, base_settings):
        """
        Create AWS ECS task scheduler class for QUEENS.

        Args:
            base_settings (dict): dictionary containing settings from base class for
                                  further use and completion in this child class 

        Returns:
            scheduler (obj):      instance of scheduler class

        """
        # get input options for scheduler from base settings
        scheduler_input_options = base_settings['scheduler_input_options']

        # initalize sub-dictionary for cluster options within base settings to None
        base_settings['cluster_options'] = {}
        base_settings['cluster_options']['singularity_path'] = None
        base_settings['cluster_options']['singularity_bind'] = None

        # initalize sub-dictionary for ECS task options within base settings
        base_settings['ecs_task_options'] = {}

        # get Docker image
        base_settings['ecs_task_options']['docker_image'] = scheduler_input_options['docker_image']

        # set absolute path to taskscript template
        script_dir = os.path.dirname(__file__)  # absolute path to directory of this file
        rel_path = '../utils/regtask_aws_docker_queens.json'
        abs_path = os.path.join(script_dir, rel_path)
        base_settings['ecs_task_options']['taskscript_template'] = abs_path

        return cls(base_settings)

    # ------------------- CHILD METHODS THAT MUST BE IMPLEMENTED ------------------
    def pre_run(self):
        """
        Pre-run routine for ECS task scheduler: check whether new task definition
        is required

        Returns:
            None

        """

        # check image and container path in most recent revision of task
        # definition 'docker-qstdoutueens', if existent
        new_task_definition_required = False
        cmd = 'aws ecs describe-task-definition --task-definition docker-queens'
        _, _, stdout, stderr = run_subprocess(cmd)

        # new task definition required, since definition 'docker-queens' is not existent
        if stderr:
            new_task_definition_required = True

        # check image and container path in most recent revision of task
        # definition 'docker-queens'
        image_str = extract_string_from_output("image", stdout)
        container_path_str = extract_string_from_output("containerPath", stdout)

        # new task definition required, since definition 'docker-queens'
        # is not equal to desired one
        if image_str != str(self.ecs_task_options['docker_image']) or container_path_str != str(
            self.experiment_dir
        ):
            new_task_definition_required = True

        # submit new task definition
        if new_task_definition_required:
            # set data for task definition
            self.ecs_task_options['EXPDIR'] = str(self.experiment_dir)
            self.ecs_task_options['IMAGE'] = str(self.ecs_task_options['docker_image'])

            # Parse data to submission script template
            submission_script_path = os.path.join(self.experiment_dir, 'taskscript.json')
            generate_submission_script(
                self.ecs_task_options,
                submission_script_path,
                self.ecs_task_options['taskscript_template'],
            )

            # change directory
            os.chdir(self.experiment_dir)

            # register task definition
            cmdlist = [
                "aws ecs register-task-definition --cli-input-json file://",
                submission_script_path,
            ]
            cmd = ''.join(cmdlist)
            _, _, _, stderr = run_subprocess(cmd)

            # detection of failed command
            if stderr:
                raise RuntimeError(
                    "\nThe task definition could not be registered properly!" f"\nStderr:\n{stderr}"
                )

    def _submit_singularity(self):
        """
        Not possible for ECS task scheduler: throw error message

        Returns:
            None

        """
        raise ValueError("\nSingularity cannot be used with ECS task scheduling!")

    def check_job_completion(self, job):
        """
        Check whether this job has been completed

        Returns:
            None

        """
        # intialize completion and failure flags to false
        completed = False
        failed = False

        command_list = [
            "aws ecs describe-tasks ",
            "--cluster worker-queens-cluster ",
            "--tasks ",
            job['aws_arn'],
        ]
        cmd = ''.join(filter(None, command_list))
        _, _, stdout, stderr = run_subprocess(cmd)

        # set job to failed if there is an error message
        if stderr:
            failed = True
        else:
            # extract status string from ECS task output
            status_str = extract_string_from_output("lastStatus", stdout)
            if status_str == 'STOPPED':
                completed = True

        return completed, failed

    def post_run(self):
        """
        Not necessary for AWS ECS scheduler but mandatory for parent class initialization

        Returns:
            None

        """
        pass
