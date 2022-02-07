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
        do_postprocessing (str):   string for identifying either local post-processing
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
        do_postprocessing,
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
            do_postprocessing (str):   string for identifying either local post-processing
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
        self.do_postprocessing = do_postprocessing
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

    @classmethod
    def from_config_create_driver(
        cls,
        config,
        job_id,
        batch,
        driver_name,
        workdir=None,
        cluster_options=None,
    ):
        """Create driver from problem description.

        Args:
            config (dict):  Dictionary containing configuration from QUEENS input file
            job_id (int):   Job ID as provided in database within range [1, n_jobs]
            batch (int):    Job batch number (multiple batches possible)
            workdir (str):  Path to working directory on remote resource
            driver_name (str): Name of driver instance that should be realized

        Returns:
            driver (obj):   Driver object
        """
        from pqueens.drivers.baci_driver import BaciDriver

        # determine Driver class
        driver_dict = {
            'baci': BaciDriver,
        }
        if driver_name:
            driver_version = config[driver_name]['driver_type']
        else:
            driver_version = config['driver']['driver_type']

        driver_class = driver_dict[driver_version]
        driver = driver_class.from_config_create_driver(
            config,
            job_id,
            batch,
            driver_name,
            abs_path,
            workdir,
            cluster_options,
        )

        return driver

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
        """Post-process, post-post process and finalize job in database."""
        if self.do_postprocessing:
            self.postprocess_job()
        if self.do_postpostprocessing:
            self.postpostprocessing()
        else:
            # set result to "no" and load job from database, if there
            # has not been any post-post-processing before
            self.result = 'no post-post-processed result'
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

    def postpostprocessing(self):
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
            # call post-post-processing
            self.result = self.postpostprocessor.get_data_from_post_file(self.output_directory)

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
    def postprocess_job(self):
        """Abstract method for post-processing of job.

        Returns:
            None
        """
        pass
