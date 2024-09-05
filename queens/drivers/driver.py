"""QUEENS driver module base class."""

import abc
import logging

_logger = logging.getLogger(__name__)


class Driver(metaclass=abc.ABCMeta):
    """Abstract base class for drivers in QUEENS.

    Attributes:
        parameters (Parameters): Parameters object
        files_to_copy (list): files or directories to copy to experiment_dir
    """

    def __init__(self, parameters, files_to_copy=None):
        """Initialize Driver object.

        Args:
            parameters (Parameters): Parameters object
            files_to_copy (list): files or directories to copy to experiment_dir
        """
        self.parameters = parameters
        if files_to_copy is None:
            files_to_copy = []
        self.files_to_copy = files_to_copy

    @abc.abstractmethod
    def run(self, sample, job_id, num_procs, experiment_dir, experiment_name):
        """Abstract method for driver run.

        Args:
            sample (dict): Dict containing sample
            job_id (int): Job ID
            num_procs (int): number of processors
            experiment_name (str): name of QUEENS experiment.
            experiment_dir (Path): Path to QUEENS experiment directory.

        Returns:
            Result and potentially the gradient
        """
