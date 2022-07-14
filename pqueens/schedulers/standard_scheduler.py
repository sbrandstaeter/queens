"""Standard scheduler for QUEENS runs."""
import logging
import multiprocessing as mp
from multiprocessing import Pool

from pqueens.drivers import from_config_create_driver
from pqueens.utils.information_output import print_scheduling_information
from pqueens.utils.manage_singularity import _check_if_new_image_needed, create_singularity_image
from pqueens.utils.path_utils import relative_path_from_queens
from pqueens.utils.run_subprocess import run_subprocess
from pqueens.utils.string_extractor_and_checker import check_if_string_in_file

from .scheduler import Scheduler

_logger = logging.getLogger(__name__)


class StandardScheduler(Scheduler):
    """Standard scheduler class for QUEENS.

    Args:
        experiment_name (str):     name of QUEENS experiment
        input_file (str):          path to QUEENS input file
        experiment_dir (str):      path to QUEENS experiment directory
        driver_name (str):         Name of the driver that shall be used for job submission
        config (dict):             dictionary containing configuration as provided in
                                   QUEENS input file
        cluster_options (dict):    (only for cluster schedulers Slurm and PBS) further
                                   cluster options
        remote (bool):             flag for remote scheduling
        remote connect (str):      (only for remote scheduling) address of remote
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
            experiment_dir (str):      path to QUEENS experiment directory
            driver_name (str):         Name of the driver that shall be used for job submission
            config (dict):             dictionary containing configuration as provided in
                                       QUEENS input file
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
        singularity = scheduler_options.get('singularity', False)
        if not isinstance(singularity, bool):
            raise TypeError(
                f"The option 'singularity' in the scheduler part of the input file has to be a"
                f" boolean, however you provided '{singularity}' which is of type "
                f"{type(singularity)} "
            )
        if singularity:
            if _check_if_new_image_needed():
                _logger.info("Local singularity image is outdated/missing, building a new one!")
                create_singularity_image()
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

    def _submit_singularity(self, job_id, batch):
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
        local_singularity_path = relative_path_from_queens("singularity_image.sif")

        cmdlist_remote_main = [
            'singularity run',
            local_singularity_path,
            remote_args,
        ]
        cmd_remote_main = ' '.join(cmdlist_remote_main)
        print(cmd_remote_main)
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
        # initialize completion and failure flags to false
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

    def _submit_driver(self, job_id, batch):
        """Submit job to driver.

        Args:
            job_id (int):    ID of job to submit
            batch (str):     Batch number of job

        Returns:
            driver_obj.pid (int): process ID
        """
        # create driver
        # TODO we should not create the object here everytime!
        # TODO instead only update the attributes of the instance.
        # TODO we should specify the data base sheet as well
        driver_obj = from_config_create_driver(
            self.config, job_id, batch, self.driver_name, cluster_options=self.cluster_options
        )

        # run driver and get process ID
        driver_obj.pre_job_run_and_run_job()
        pid = driver_obj.pid

        # only required for standard scheduling: finish-and-clean call
        # (taken care of by submit_data_processor for other schedulers)
        if self.scheduler_type == 'standard':
            driver_obj.post_job_run()

        return pid
