"""Standard scheduler for QUEENS runs."""
import os
import sys

from pqueens.utils.information_output import print_driver_information, print_scheduling_information
from pqueens.utils.run_subprocess import run_subprocess
from pqueens.utils.string_extractor_and_checker import check_if_string_in_file

from .scheduler_new import Scheduler


class StandardScheduler(Scheduler):
    """Standard scheduler class for QUEENS.

    Args:
        experiment_name (str):     name of QUEENS experiment
        input_file (str):          path to QUEENS input file
        restart (bool):            flag for restart
        experiment_dir (str):      path to QUEENS experiment directory
        driver_name (str):         Name of the driver that shall be used for job submission
        config (dict):             dictionary containing configuration as provided in
                                   QUEENS input file
        restart (bool):            flag for restart
        cluster_options (dict):    (only for cluster schedulers Slurm and PBS) further
                                   cluster options
        remote (bool):             flag for remote scheduling
        remote connect (str):      (only for remote scheduling) adress of remote
                                   computing resource
        port (int):                (only for remote scheduling with Singularity) port of
                                   remote resource for ssh port-forwarding to database
        cluster_options (dict):    (only for cluster schedulers Slurm and PBS) further
                                   cluster options
        singularity (bool):        flag for use of Singularity containers
        scheduler_type (str):      type of scheduler chosen in QUEENS input file
        process_ids (dict): Dict of process-IDs of the submitted process as value with job_ids as
                           keys
    """

    def __init__(
        self,
        experiment_name,
        input_file,
        restart,
        experiment_dir,
        driver_name,
        config,
        cluster_options,
        singularity,
        scheduler_type,
    ):
        """Initialize Standard scheduler.

        Args:
            experiment_name (str):     name of QUEENS experiment
            input_file (str):          path to QUEENS input file
            restart (bool):            flag for restart
            experiment_dir (str):      path to QUEENS experiment directory
            driver_name (str):         Name of the driver that shall be used for job submission
            config (dict):             dictionary containing configuration as provided in
                                       QUEENS input file
            restart (bool):            flag for restart
            cluster_options (dict):    (only for cluster schedulers Slurm and PBS) further
                                       cluster options
            remote (bool):             flag for remote scheduling
            remote connect (str):      (only for remote scheduling) adress of remote
                                       computing resource
            port (int):                (only for remote scheduling with Singularity) port of
                                       remote resource for ssh port-forwarding to database
            cluster_options (dict):    (only for cluster schedulers Slurm and PBS) further
                                       cluster options
            singularity (bool):        flag for use of Singularity containers
            scheduler_type (str):      type of scheduler chosen in QUEENS input file
            process_ids (dict): Dict of process-IDs of the submitted process as value with job_ids
                                as keys
        """
        super().__init__(
            experiment_name,
            input_file,
            restart,
            experiment_dir,
            driver_name,
            config,
            cluster_options,
            singularity,
            scheduler_type,
        )

    @classmethod
    def from_config_create_scheduler(cls, config, scheduler_name, driver_name):
        """Create standard scheduler object from config.

        Args:
            config (dict): QUEENS input dictionary
            scheduler_name (str): Name of the scheduler
            driver_name (str): Name of the driver

        Returns:
            instance of standard scheduler class
        """
        if not scheduler_name:
            scheduler_name = "scheduler"
        scheduler_options = config[scheduler_name]

        if scheduler_options.get("remote", False):
            raise NotImplementedError("Standard scheduler can not be used remotely")

        experiment_name = config['global_settings']['experiment_name']
        experiment_dir = scheduler_options['experiment_dir']
        input_file = config["input_file"]
        restart = config.get("restart", False)
        singularity = scheduler_options.get('singularity', False)
        cluster_options = {}
        scheduler_type = scheduler_options["scheduler_type"]

        # TODO move this to a different place
        # print out scheduling information
        print_scheduling_information(
            scheduler_type,
            False,
            None,
            singularity,
        )
        return cls(
            experiment_name,
            input_file,
            restart,
            experiment_dir,
            driver_name,
            config,
            cluster_options,
            singularity,
            scheduler_type,
        )

    # ------------------- CHILD METHODS THAT MUST BE IMPLEMENTED ------------------
    def pre_run(self):
        """Pre-run routine.

        Not needed as here no remote operations are done.
        """
        pass

    def _submit_singularity(self, job_id, batch, restart):
        """Submit job locally to Singularity.

        Args:
            job_id (int):    ID of job to submit
            batch (str):     Batch number of job

        Returns:
            int:            process ID
        """
        local_path_json = self.input_file
        remote_args = '--job_id={} --batch={} --port={} --path_json={} --driver_name={}'.format(
            job_id, batch, '000', local_path_json, self.driver_name
        )
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        rel_path = '../../singularity_image.sif'
        local_singularity_path = os.path.join(script_dir, rel_path)
        if restart:
            cmdlist_remote_main = [
                '/usr/bin/singularity run',
                local_singularity_path,
                remote_args,
                '--post=true',
            ]
            cmd_remote_main = ' '.join(cmdlist_remote_main)
            run_subprocess(cmd_remote_main)
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

    def check_job_completion(self, job):
        """Check whether this job has been completed.

        Args:
            job (dict): Job dict.

        Returns:
            completed (bool): If job is completed
            failed (bool): If job failed.
        """
        # intialize completion and failure flags to false
        # (Note that failure is not checked for standard scheduler
        #  and returned false in any case.)
        failed = False

        # set string to search for in output file
        search_string = 'Total CPU Time for CALCULATION'
        # indicate completion by existing search string in local output file
        completed = check_if_string_in_file(job['log_file_path'], search_string)
        return completed, failed

    def post_run(self):
        """Post run routines."""
        pass
