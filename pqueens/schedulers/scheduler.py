"""QUEENS scheduler parent class."""
import abc

from pqueens.drivers import from_config_create_driver


class Scheduler(metaclass=abc.ABCMeta):
    """Abstract base class for schedulers in QUEENS.

    The scheduler manages simulation runs in QUEENS on local or remote
    computing resources with or without Singularity containers using
    various scheduling systems on respective computing resource (see
    also respective Wiki article for more details).

    Attributes:
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
            process_ids (dict): Dict of process-IDs of the submitted process as value with job_ids
                                as keys
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
        """Initialise Scheduler.

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

        Returns:
            scheduler (obj):           instance of scheduler class
        """
        self.experiment_name = experiment_name
        self.input_file = input_file
        self.experiment_dir = experiment_dir
        self.driver_name = driver_name
        self.config = config
        self.cluster_options = cluster_options
        self.scheduler_type = scheduler_type
        self.singularity = singularity
        self.process_ids = {}

    # ------------------------ AUXILIARY HIGH LEVEL METHODS -----------------------
    def submit(self, job_id, batch):
        """Function to submit job to scheduling software on a resource.

        Args:
            job_id (int):            ID of job to submit
            batch (int):             Batch number of job

        Returns:
            pid (int):               process id of job
        """
        if self.singularity:
            pid = self._submit_singularity(job_id, batch)
        else:
            pid = self._submit_driver(job_id, batch)

        self.process_ids[str(job_id)] = pid

        return pid

    def submit_data_processor(self, job_id, batch):
        """Function to submit data processor job to scheduling software.

        Args:
            job_id (int):            ID of job to submit
            batch (int):             Batch number of job
        """
        # create driver
        # TODO we should not create a new driver instance here every time
        # instead only update the driver attributes.
        driver_obj = from_config_create_driver(self.config, job_id, batch, self.driver_name)

        # do post-processing (if required), data-processing,
        # finish and clean job
        driver_obj.post_job_run()

    # ------- CHILDREN METHODS THAT NEED TO BE IMPLEMENTED / ABSTRACTMETHODS ------
    @abc.abstractmethod
    def pre_run(self):
        """Pre run routine."""
        pass

    @abc.abstractmethod
    def _submit_singularity(self, job_id, batch):
        """Submit job using singularity."""
        pass

    @abc.abstractmethod
    def check_job_completion(self, job):
        """Check whether this job has been completed.

        Args:
            job (dict): Job dict.

        Returns:
            completed (bool): If job is completed
            failed (bool): If job failed.
        """
        pass

    @abc.abstractmethod
    def post_run(self):
        """Post run routine."""
        pass

    @abc.abstractmethod
    def _submit_driver(self, job_id, batch):
        """Submit job to driver."""
        pass
