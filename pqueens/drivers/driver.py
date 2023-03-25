"""QUEENS driver module base class."""
import abc
import logging
import time

_logger = logging.getLogger(__name__)


class Driver(metaclass=abc.ABCMeta):
    """Abstract base class for drivers in QUEENS.

    The driver manages simulation runs in QUEENS on local or remote computing resources,
    with or without Singularity containers, depending on the chosen CAE software (see also
    respective Wiki article on available CAE software).

    Attributes:
        batch (int):               Current batch of driver calls.
        driver_name (str):         Name of the driver used for the analysis. The name is
                                   specified in the json-input file.
        experiment_dir (Path):     Path to QUEENS experiment directory.
        experiment_name (str):     Name of QUEENS experiment.
        job (dict, None):          Dictionary containing description of current job.
        job_id (int):              Job ID as provided in database within range [1, n_jobs].
        num_procs (int):           Number of processors for processing.
        output_directory (Path):   Path to output directory (on remote computing resource for
                                   remote scheduling).
        result (np.array):         Simulation result to be stored in database.
        gradient (np.array):       Gradient of the simulation output w.r.t. the input.
        database (obj):            Database object.
        post_processor (Path):     (only for post-processing) Path to *post_processor* of
                                   respective CAE software.
        gradient_data_processor (obj): Instance of data processor class for gradient data.
        data_processor (obj):   Instance of data processor class.
    """

    def __init__(
        self,
        batch,
        driver_name,
        experiment_dir,
        experiment_name,
        job,
        job_id,
        num_procs,
        output_directory,
        result,
        gradient,
        database,
        post_processor,
        gradient_data_processor,
        data_processor,
    ):
        """Initialize driver obj.

        Args:
            batch (int):               Current batch of driver calls.
            driver_name (str):         Name of the driver used for the analysis. The name is
                                       specified in the json-input file.
            experiment_dir (Path):     Path to QUEENS experiment directory
            experiment_name (str):     Name of QUEENS experiment
            job (dict,None):           Dictionary containing description of current job
            job_id (int):              Job ID as provided in database within range [1, n_jobs]
            num_procs (int):           Number of processors for processing
            output_directory (Path):   Path to output directory (on remote computing resource for
                                       remote scheduling)
            result (np.array):         Simulation result to be stored in database
            gradient (np.array): Gradient of the simulation output w.r.t. to the input
            database (obj):            Database object
            post_processor (Path):     (Only for post-processing) path to post_processor of
                                       respective CAE software
            data_processor (obj):      Instance of data processor class
            gradient_data_processor (obj):   Instance of data processor class for gradient data
        """
        self.batch = batch
        self.driver_name = driver_name
        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_name
        self.job = job
        self.job_id = job_id
        self.num_procs = num_procs
        self.output_directory = output_directory
        self.result = result
        self.gradient = gradient
        self.database = database
        self.post_processor = post_processor
        self.gradient_data_processor = gradient_data_processor
        self.data_processor = data_processor

    # ------ Core methods ----------------------------------------------------- #
    def pre_job_run_and_run_job(self):
        """Prepare and execute job run."""
        self.pre_job_run()
        self.run_job()

    def pre_job_run(self):
        """Prepare job run."""
        if self.job is None:
            self.initialize_job_in_db()
        self.prepare_input_files()

    def post_job_run(self):
        """Post-process, data processing and finalize job in database."""
        if self.post_processor:
            self.post_processor_job()

        if self.gradient_data_processor:
            self.gradient_data_processor_job()

        if self.data_processor:
            self.data_processor_job()
        else:
            # set result to "no" and load job from database, if there
            # has not been any data-processing before
            self.result = 'no processed data result'
            if self.job is None:
                self.job = self._load_job_from_db()

        self.finalize_job_in_db()

    # ------ Base class methods ------------------------------------------------ #
    def initialize_job_in_db(self):
        """Initialize job in database."""
        self.job = self._load_job_from_db()

        start_time = time.time()
        self.job['start_time'] = start_time
        self._save_job_in_db()

    def data_processor_job(self):
        """Extract data of interest from post-processed file.

        Afterwards save them to the database.
        """
        # load job from database if existent
        if self.job is None:
            self.job = self._load_job_from_db()
        # only proceed if this job did not fail
        if self.job['status'] != "failed":
            self.result = self.data_processor.get_data_from_file(self.output_directory)
            _logger.debug("Got result: %s", self.result)

    def gradient_data_processor_job(self):
        """Extract gradient data from post-processed file.

        Afterwards save them to the database.
        """
        # load job from database if existent
        if self.job is None:
            self.job = self._load_job_from_db()
        # only proceed if this job did not fail
        if self.job['status'] != "failed":
            self.gradient = self.gradient_data_processor.get_data_from_file(self.output_directory)
            _logger.debug("Got gradient: %s", self.gradient)

    def finalize_job_in_db(self):
        """Finalize job in database."""
        if self.result is None:
            self.job['result'] = None
            self.job['gradient'] = None
            self.job['status'] = 'failed'
            self.job['end_time'] = time.time()
            self._save_job_in_db()
        else:
            self.job['result'] = self.result
            self.job['gradient'] = self.gradient
            self.job['status'] = 'complete'
            if self.job['start_time']:
                self.job['end_time'] = time.time()
                computing_time = self.job['end_time'] - self.job['start_time']
                _logger.info(
                    "Successfully completed job %s (No. of proc.: %s, computing time: %s s).\n",
                    self.job_id,
                    self.num_procs,
                    computing_time,
                )
            self._save_job_in_db()
            _logger.info("Saved job %s to database.", self.job_id)

    def _load_job_from_db(self):
        """Load job from database."""
        return self.database.load(
            self.experiment_name,
            self.batch,
            'jobs_' + self.driver_name,
            {'id': self.job_id},
        )

    def _save_job_in_db(self):
        """Save job in database."""
        self.database.save(
            self.job,
            self.experiment_name,
            'jobs_' + self.driver_name,
            str(self.batch),
            {
                'id': self.job_id,
                'experiment_dir': str(self.experiment_dir),
                'experiment_name': self.experiment_name,
            },
        )

    @abc.abstractmethod
    def prepare_input_files(self):
        """Abstract method for preparing input file(s)."""

    @abc.abstractmethod
    def run_job(self):
        """Abstract method for running job."""

    @abc.abstractmethod
    def post_processor_job(self):
        """Abstract method for post-processing of job."""
