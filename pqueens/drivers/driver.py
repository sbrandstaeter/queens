"""QUEENS driver module base class."""
import abc
import logging
import time

_logger = logging.getLogger(__name__)


class Driver(metaclass=abc.ABCMeta):
    """Abstract base class for drivers in QUEENS.

    The driver manages simulation runs in QUEENS on local or remote computing resources
    with or without Singularity containers depending on the chosen CAE software (see also
    respective Wiki article on available CAE software).

    Attributes:
        batch (int):               Current batch of driver calls.
        direct_scheduling(bool):   flag for direct scheduling
        post_processor_location (str):   string for identifying either local post-processing
                                   ('local') or remote post-processing ('remote') or 'None'
        driver_name (str):         Name of the driver used for the analysis. The name is
                                   specified in the json-input file.
        experiment_dir (str):      path to QUEENS experiment directory
        experiment_name (str):     name of QUEENS experiment
        job (dict):                dictionary containing description of current job
        job_id (int):              job ID as provided in database within range [1, n_jobs]
        num_procs (int):           number of processors for processing
        output_directory (str):    path to output directory (on remote computing resource for
                                   remote scheduling)
        remote (bool):             flag for remote scheduling
        remote_connect (str):      (only for remote scheduling) adress of remote
                                   computing resource
        result (np.array):         simulation result to be stored in database
        singularity (bool):        flag for use of Singularity containers
        database (obj):            database object
    """

    def __init__(
        self,
        batch,
        direct_scheduling,
        post_processor_location,
        driver_name,
        experiment_dir,
        experiment_name,
        job,
        job_id,
        num_procs,
        output_directory,
        remote,
        remote_connect,
        result,
        singularity,
        database,
    ):
        """Initialize driver obj.

        Args:
            batch (int):               Current batch of driver calls.
            direct_scheduling(bool):   flag for direct scheduling
            post_processor_location (str):   string for identifying either local post-processing
                                       ('local') or remote post-processing ('remote') or 'None'
            driver_name (str):         Name of the driver used for the analysis. The name is
                                       specified in the json-input file.
            experiment_dir (str):      path to QUEENS experiment directory
            experiment_name (str):     name of QUEENS experiment
            job (dict):                dictionary containing description of current job
            job_id (int):              job ID as provided in database within range [1, n_jobs]
            num_procs (int):           number of processors for processing
            output_directory (str):    path to output directory (on remote computing resource for
                                       remote scheduling)
            remote (bool):             flag for remote scheduling
            remote_connect (str):      (only for remote scheduling) adress of remote
                                       computing resource
            result (np.array):         simulation result to be stored in database
            singularity (bool):        flag for use of Singularity containers
            database (obj):            database object
        """
        self.batch = batch
        self.direct_scheduling = direct_scheduling
        self.post_processor_location = post_processor_location
        self.driver_name = driver_name
        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_name
        self.job = job
        self.job_id = job_id
        self.num_procs = num_procs
        self.output_directory = output_directory
        self.remote = remote
        self.remote_connect = remote_connect
        self.result = result
        self.singularity = singularity
        self.database = database

    # ------ Core methods ----------------------------------------------------- #
    def pre_job_run_and_run_job(self):
        """Prepare and execute job run.

        Returns:
            None
        """
        self.pre_job_run()
        self.run_job()

    def pre_job_run(self):
        """Prepare job run.

        Returns:
            None
        """
        if self.job is None:
            self.initialize_job_in_db()
        self.prepare_input_files()

    def post_job_run(self):
        """Post-process, data processing and finalize job in database."""
        if self.post_processor:
            self.post_processor_job()
        if self.data_processor:
            self.data_processor_job()
        else:
            # set result to "no" and load job from database, if there
            # has not been any data-processing before
            self.result = 'no processed data result'
            if self.job is None:
                self.job = self.database.load(
                    self.experiment_name,
                    self.batch,
                    'jobs_' + self.driver_name,
                    {'id': self.job_id},
                )

        self.finalize_job_in_db()

    # ------ Base class methods ------------------------------------------------ #
    def initialize_job_in_db(self):
        """Initialize job in database.

        Returns:
            None
        """
        # load job from database
        self.job = self.database.load(
            self.experiment_name,
            self.batch,
            'jobs_' + self.driver_name,
            {'id': self.job_id, 'expt_dir': self.experiment_dir, 'expt_name': self.experiment_name},
        )

        # set start time and store it in database
        start_time = time.time()
        self.job['start time'] = start_time

        # save start time in database to make it accessible for the second post-processing call
        self.database.save(
            self.job,
            self.experiment_name,
            'jobs_' + self.driver_name,
            str(self.batch),
            {'id': self.job_id, 'expt_dir': self.experiment_dir, 'expt_name': self.experiment_name},
        )

    def data_processor_job(self):
        """Extract data of interest from post-processed file.

        Afterwards save them to the database.
        """
        # load job from database if existent
        if self.job is None:
            self.job = self.database.load(
                self.experiment_name,
                self.batch,
                'jobs_' + self.driver_name,
                {'id': self.job_id},
            )
        # only proceed if this job did not fail
        if self.job['status'] != "failed":
            # call data-processing
            self.result = self.data_processor.get_data_from_file(self.output_directory)

            _logger.info(f"Got result: {self.result}")

    def finalize_job_in_db(self):
        """Finalize job in database."""
        if self.result is None:
            self.job['result'] = None  # TODO: maybe we should better use a pandas format here
            self.job['status'] = 'failed'
            if not self.direct_scheduling:
                self.job['end time'] = time.time()
            self.database.save(
                self.job,
                self.experiment_name,
                'jobs_' + self.driver_name,
                str(self.batch),
                {
                    'id': self.job_id,
                    'expt_dir': self.experiment_dir,
                    'expt_name': self.experiment_name,
                },
            )
        else:
            self.job['result'] = self.result
            self.job['status'] = 'complete'
            if self.job['start time'] and not self.direct_scheduling:
                self.job['end time'] = time.time()
                computing_time = self.job['end time'] - self.job['start time']
                _logger.info(
                    "Successfully completed job {:d} (No. of proc.: {:d}, "
                    "computing time: {:08.2f} s).\n".format(
                        self.job_id, self.num_procs, computing_time
                    )
                )
            self.database.save(
                self.job,
                self.experiment_name,
                'jobs_' + self.driver_name,
                str(self.batch),
                {
                    'id': self.job_id,
                    'expt_dir': self.experiment_dir,
                    'expt_name': self.experiment_name,
                },
            )

    # ---------------- COMMAND-ASSEMBLY METHODS ----------------------------------
    def assemble_remote_run_cmd(self, run_cmd):
        """Assemble command for remote run.

        Returns:
            remote run command
        """
        command_list = [
            'ssh',
            self.remote_connect,
            '"cd',
            self.experiment_dir,
            ';',
            run_cmd,
            '"',
        ]

        return ' '.join(filter(None, command_list))

    # ---------------- CHILD METHODS THAT NEED TO BE IMPLEMENTED ---------------
    @abc.abstractmethod
    def prepare_input_files(self):
        """Abstract method for preparing input file(s)."""
        pass

    @abc.abstractmethod
    def run_job(self):
        """Abstract method for running job.

        Returns:
            None
        """
        pass

    @abc.abstractmethod
    def post_processor_job(self):
        """Abstract method for post-processing of job.

        Returns:
            None
        """
        pass
