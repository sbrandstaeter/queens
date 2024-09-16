"""Pool scheduler for QUEENS runs."""

import logging
from functools import partial

import numpy as np
from tqdm import tqdm

from queens.schedulers.scheduler import Scheduler
from queens.utils.config_directories import experiment_directory
from queens.utils.logger_settings import log_init_args
from queens.utils.pool_utils import create_pool

_logger = logging.getLogger(__name__)


class PoolScheduler(Scheduler):
    """Pool scheduler class for QUEENS.

    Attributes:
        pool (pathos pool): Multiprocessing pool.
        verbose (bool): Verbosity of evaluations.
    """

    @log_init_args
    def __init__(self, experiment_name, num_jobs=1, verbose=True):
        """Initialize PoolScheduler.

        Args:
            experiment_name (str): name of the current experiment
            num_jobs (int, opt): Maximum number of parallel jobs
            verbose (bool, opt): verbosity of evaluations
        """
        super().__init__(
            experiment_name=experiment_name,
            experiment_dir=experiment_directory(experiment_name=experiment_name),
            num_jobs=num_jobs,
        )
        self.pool = create_pool(num_jobs)
        self.verbose = verbose

    def evaluate(self, samples, driver, job_ids=None):
        """Submit jobs to driver.

        Args:
            samples (np.array): Array of samples
            driver (Driver): Driver object that runs simulation
            job_ids (lst, opt): List of job IDs corresponding to samples

        Returns:
            result_dict (dict): Dictionary containing results
        """
        function = partial(
            driver.run,
            num_procs=1,
            experiment_dir=self.experiment_dir,
            experiment_name=self.experiment_name,
        )
        if job_ids is None:
            job_ids = self.get_job_ids(len(samples))
        # Pool or no pool
        if self.pool:
            results = self.pool.map(function, samples, job_ids)
        elif self.verbose:
            results = list(map(function, tqdm(samples), job_ids))
        else:
            results = list(map(function, samples, job_ids))

        output = {}
        # check if gradient is returned --> tuple
        if isinstance(results[0], tuple):
            results_iterator, gradient_iterator = zip(*results)
            results_array = np.array(list(results_iterator))
            gradients_array = np.array(list(gradient_iterator))
            output["gradient"] = gradients_array
        else:
            results_array = np.array(results)

        output["result"] = results_array
        return output
