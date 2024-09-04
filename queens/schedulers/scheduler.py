"""QUEENS scheduler parent class."""

import abc
import logging

from queens.utils.rsync import rsync

_logger = logging.getLogger(__name__)


class Scheduler(metaclass=abc.ABCMeta):
    """Abstract base class for schedulers in QUEENS.

    Attributes:
        experiment_name (str): name of the current experiment
        experiment_dir (Path): Path to QUEENS experiment directory.
    """

    def __init__(self, experiment_name, experiment_dir):
        """Initialize scheduler.

        Args:
            experiment_name (str): name of QUEENS experiment.
            experiment_dir (Path): Path to QUEENS experiment directory.
        """
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir

    @abc.abstractmethod
    def evaluate(self, samples_list, driver):
        """Submit jobs to driver.

        Args:
            samples_list (list): List of dicts containing samples and job ids
            driver (Driver): Driver object that runs simulation

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
