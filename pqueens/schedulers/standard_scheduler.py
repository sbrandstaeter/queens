import os
import sys
from .scheduler import Scheduler
from pqueens.utils.run_subprocess import run_subprocess
from pqueens.utils.string_extractor_and_checker import check_if_string_in_file


class StandardScheduler(Scheduler):
    """
    Standard/nohup scheduler class for QUEENS.

    """

    def __init__(self, base_settings):
        super(StandardScheduler, self).__init__(base_settings)

    @classmethod
    def create_scheduler_class(cls, base_settings):
        """
        Create standard/nohup scheduler class for QUEENS.

        Args:
            base_settings (dict): dictionary containing settings from base class for
                                  further use and completion in this child class 

        Returns:
            scheduler (obj):      instance of scheduler class

        """
        # initalize sub-dictionary for cluster and ECS task options
        # within base settings to None
        base_settings['cluster_options'] = {}
        base_settings['cluster_options']['singularity_path'] = None
        base_settings['cluster_options']['singularity_bind'] = None
        base_settings['ecs_task_options'] = None

        return cls(base_settings)

    # ------------------- CHILD METHODS THAT MUST BE IMPLEMENTED ------------------
    def pre_run(self):
        """
        Pre-run routine for remote nohup computing: copying checker file to remote
        machine

        Returns:
            None

        """

        if self.remote and self.scheduler_type == 'nohup':
            # generate command for copying 'string_extractor_and_checker.py' to
            # experiment directory on remote machine
            checker_filename = 'string_extractor_and_checker.py'
            this_dir = os.path.dirname(__file__)
            rel_path = os.path.join('../utils', checker_filename)
            abs_path = os.path.join(this_dir, rel_path)
            command_list = [
                "scp ",
                abs_path,
                " ",
                self.remote_connect,
                ":",
                self.experiment_dir,
            ]
            command_string = ''.join(command_list)
            _, _, _, stderr = run_subprocess(command_string)

            # detection of failed command
            if stderr:
                raise RuntimeError(
                    "\nString checker file could not be copied to remote machine!"
                    f"\nStderr:\n{stderr}"
                )

    def _submit_singularity(self, job_id, batch, restart):
        """Submit job locally to Singularity

        Args:
            job_id (int):    ID of job to submit
            batch (str):     Batch number of job

        Returns:
            int:            process ID

        """
        if not self.remote and self.scheduler_type == 'standard':
            local_path_json = self.input_file
            remote_args = '--job_id={} --batch={} --port={} --path_json={} --driver_name={}'.format(
                job_id, batch, '000', local_path_json, self.driver_name
            )
            script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
            rel_path = '../../driver.simg'
            local_singularity_path = os.path.join(script_dir, rel_path)
            if restart:
                cmdlist_remote_main = [
                    '/usr/bin/singularity run',
                    local_singularity_path,
                    remote_args,
                    '--post=true',
                ]
                cmd_remote_main = ' '.join(cmdlist_remote_main)
                _, _, _, _ = run_subprocess(cmd_remote_main)
                return 0
            else:
                cmdlist_remote_main = [
                    '/usr/bin/singularity run',
                    local_singularity_path,
                    remote_args,
                ]
                cmd_remote_main = ' '.join(cmdlist_remote_main)
                _, pid, _, _ = run_subprocess(cmd_remote_main, subprocess_type='submit')
                return pid
        else:
            if self.scheduler_type == 'nohup':
                raise ValueError(
                    "\nSingularity is not recommended to be used with nohup scheduling!"
                )
            else:
                raise ValueError(
                    "\nSingularity cannot yet be used remotely with standard scheduling!"
                )
            return None

    def get_cluster_job_id(self):
        """
        Not necessary for standard scheduler but mandatory for parent class initialization

        Returns:
            None

        """
        pass

    def alive(self, process_id):
        """
        Not necessary for standard scheduler but mandatory for parent class initialization

        Returns:
            None

        """
        pass

    def check_job_completion(self, job):
        """
        Check whether this job has been completed

        Returns:
            None

        """
        # intialize completion and failure flags to false
        # (Note that failure is not checked for standard scheduler
        #  and returned false in any case.)
        completed = False
        failed = False

        # set string to search for in output file
        search_string = 'Total CPU Time for CALCULATION'

        if self.remote:
            # indicate completion by existing control file in remote output directory
            # for this purpose, generate command for executing
            # 'string_extractor_and_checker.py' on remote machine
            checker_filename = 'string_extractor_and_checker.py'
            checker_path_on_remote = os.path.join(self.experiment_dir, checker_filename)
            command_list = [
                "ssh ",
                self.remote_connect,
                " 'python ",
                checker_path_on_remote,
                ' ',
                job['log_file_path'],
                " \"",
                search_string,
                "\"'",
            ]
            command_string = ''.join(command_list)
            _, _, stdout, stderr = run_subprocess(command_string)

            # detection of failed command
            if stderr:
                raise RuntimeError(
                    "\nString checker file could not be executed on remote machine!"
                    f"\nStderr on remote:\n{stderr}"
                )

            # search string present
            if stdout:
                completed = True
        else:
            # indicate completion by existing search string in local output file
            completed = check_if_string_in_file(job['log_file_path'], search_string)

        return completed, failed

    def post_run(self):
        """
        Post-run routine for remote nohup computing: removing checker file from remote
        machine

        Returns:
            None

        """
        if self.remote and self.scheduler_type == 'nohup':
            # generate command for removing 'string_extractor_and_checker.py'
            # from experiment directory on remote machine
            checker_filename = 'string_extractor_and_checker.py'
            checker_path_on_remote = os.path.join(self.experiment_dir, checker_filename)
            command_list = [
                'ssh',
                self.remote_connect,
                '"rm',
                checker_path_on_remote,
                '"',
            ]
            command_string = ' '.join(command_list)
            _, _, _, stderr = run_subprocess(command_string)

            # detection of failed command
            if stderr:
                raise RuntimeError(
                    "\nChecker file could not be removed from remote machine!"
                    f"\nStderr on remote:\n{stderr}"
                )
