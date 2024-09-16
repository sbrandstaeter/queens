"""QUEENS scheduler parent class."""

import abc
import logging

import numpy as np

from queens.utils.rsync import rsync

_logger = logging.getLogger(__name__)


class Scheduler(metaclass=abc.ABCMeta):
    """Abstract base class for schedulers in QUEENS.

    Attributes:
        experiment_name (str): name of the current experiment
        experiment_dir (Path): Path to QUEENS experiment directory.
        num_jobs (int): Maximum number of parallel jobs
        latest_job_id (int):    Latest job ID.
    """

    def __init__(self, experiment_name, experiment_dir, num_jobs):
        """Initialize scheduler.

        Args:
            experiment_name (str): name of QUEENS experiment.
            experiment_dir (Path): Path to QUEENS experiment directory.
            num_jobs (int): Maximum number of parallel jobs
        """
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir
        self.num_jobs = num_jobs
        self.latest_job_id = 0

    @abc.abstractmethod
    def evaluate(self, samples, driver, job_ids=None):
        """Submit jobs to driver.

        Args:
            samples (np.array): Array of samples
            driver (Driver): Driver object that runs simulation
            job_ids (lst, opt): List of job IDs corresponding to samples

        Returns:
            result_dict (dict): Dictionary containing results
        """

    def copy_files_to_experiment_dir(self, paths):
        """Copy file to experiment directory.

        Args:
            paths (Path, list): paths to files or directories that should be copied to experiment
                                directory
        """
        destination = f"{self.experiment_dir}/"
        rsync(paths, destination)

    def get_job_ids(self, num_samples):
        """Get job ids and update latest_job_id.

        Args:
            num_samples (int): Number of samples

        Returns:
            job_ids (np.array): Array of job ids
        """
        job_ids = self.latest_job_id + np.arange(1, num_samples + 1)
        self.latest_job_id += num_samples
        return job_ids
