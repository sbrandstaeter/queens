"""Standard scheduler for QUEENS runs."""
import logging
import pathlib
from threading import Thread

from pqueens.drivers import from_config_create_driver
from pqueens.schedulers.scheduler import Scheduler
from pqueens.utils.config_directories import ABS_SINGULARITY_IMAGE_PATH, experiment_directory
from pqueens.utils.dictionary_utils import find_keys
from pqueens.utils.manage_singularity import create_singularity_image, new_singularity_image_needed
from pqueens.utils.print_utils import get_str_table
from pqueens.utils.run_subprocess import run_subprocess
from pqueens.utils.string_extractor_and_checker import check_if_string_in_file

_logger = logging.getLogger(__name__)


class StandardScheduler(Scheduler):
    """Standard scheduler class for QUEENS.

    Attributes:
        max_concurrent (int): Number of maximum jobs that run in parallel
    """

    def __init__(
        self,
        experiment_name,
        input_file,
        experiment_dir,
        driver_name,
        config,
        singularity,
        scheduler_type,
        max_concurrent,
    ):
        """Initialize Standard scheduler.

        Args:
            experiment_name (str):     name of QUEENS experiment
            input_file (path):         path to QUEENS input file
            experiment_dir (path):     path to QUEENS experiment directory
            driver_name (str):         Name of the driver that shall be used for job submission
            config (dict):             dictionary containing configuration as provided in
                                       QUEENS input file
            remote connect (str):      (only for remote scheduling) address of remote
                                       computing resource
            singularity (bool):        flag for use of Singularity containers
            scheduler_type (str):      type of scheduler chosen in QUEENS input file
            max_concurrent (int): Number of maximum jobs that run in parallel
        """
        super().__init__(
            experiment_name,
            input_file,
            experiment_dir,
            driver_name,
            config,
            singularity,
            scheduler_type,
        )
        self.max_concurrent = max_concurrent

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
        experiment_dir = experiment_directory(experiment_name=experiment_name)
        input_file = pathlib.Path(config["input_file"])
        singularity = scheduler_options.get('singularity', False)
        if not isinstance(singularity, bool):
            raise TypeError(
                f"The option 'singularity' in the scheduler part of the input file has to be a"
                f" boolean, however you provided '{singularity}' which is of type "
                f"{type(singularity)} "
            )
        if singularity:
            if new_singularity_image_needed():
                _logger.info("Local singularity image is outdated/missing, building a new one!")
                create_singularity_image()
        scheduler_type = scheduler_options["type"]

        # find the max_concurrent key in the input file
        max_concurrent_lst = list(find_keys(config, 'max_concurrent'))
        if max_concurrent_lst:
            max_concurrent = max_concurrent_lst[0]
        else:
            max_concurrent = 1

        return cls(
            experiment_name,
            input_file,
            experiment_dir,
            driver_name,
            config,
            singularity,
            scheduler_type,
            max_concurrent,
        )

    def __str__(self):
        """String description of the ClusterScheduler object.

        Returns:
            string (str): ClusterScheduler object description
        """
        name = "Standard Scheduler"
        print_dict = self._create_base_print_dict()
        return get_str_table(name, print_dict)

    # ------------------- CHILD METHODS THAT MUST BE IMPLEMENTED ------------------
    def pre_run(self):
        """Pre-run routine.

        Not needed as here no remote operations are done.
        """

    def _submit_singularity(self, job_id, batch):
        """Submit job locally to Singularity.

        Args:
            job_id (int):    ID of job to submit
            batch (str):     Batch number of job

        Returns:
            int:            process ID
        """
        remote_args = [
            f"--job_id={job_id}",
            f"--batch={batch}",
            "--port=000",
            f"--input={self.input_file}",
            f"--driver_name={self.driver_name}",
            f"--experiment_dir={self.experiment_dir}",
        ]

        cmdlist_remote_main = [f'singularity run {ABS_SINGULARITY_IMAGE_PATH}'] + remote_args
        cmd_remote_main = ' '.join(cmdlist_remote_main)
        _logger.info(cmd_remote_main)
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
        if self.max_concurrent == 1:
            # sequential scheduling
            pid = StandardScheduler.driver_execution_helper_fun(
                self.config, job_id, batch, self.driver_name, self.experiment_dir
            )
        else:
            # run the drivers in separate threads to enable parallel execution
            Thread(
                target=StandardScheduler.driver_execution_helper_fun,
                args=(self.config, job_id, batch, self.driver_name, self.experiment_dir),
            ).start()
            pid = 0

        return pid

    @staticmethod
    def driver_execution_helper_fun(config, job_id, batch, driver_name, experiment_dir):
        """Helper function to execute driver commands.

        Args:
            config (dict): Input file problem description
            job_id (int): Id number of the current job
            batch (int): Number of the current batch
            driver_name (str): Name of the driver module in input file
            experiment_dir (Path): directory of experiment

        Returns:
            pid (int): Process ID
        """
        driver_obj = from_config_create_driver(config, job_id, batch, driver_name, experiment_dir)
        # run driver and get process ID
        driver_obj.pre_job_run_and_run_job()
        pid = driver_obj.pid
        driver_obj.post_job_run()
        return pid
