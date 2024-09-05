"""QUEENS driver module base class."""

import abc
import logging

_logger = logging.getLogger(__name__)


class Driver(metaclass=abc.ABCMeta):
    """Abstract base class for drivers in QUEENS.

    Attributes:
        files_to_copy (list): files or directories to copy to experiment_dir
    """

    def __init__(self, files_to_copy=None):
        """Initialize Driver object.

        Args:
            files_to_copy (list): files or directories to copy to experiment_dir
        """
        if files_to_copy is None:
            files_to_copy = []
        self.files_to_copy = files_to_copy

    @abc.abstractmethod
    def run(self, job_id_and_sample, num_procs, experiment_dir, experiment_name):
        """Abstract method for driver run.

        Args:
            job_id_and_sample (np.array): array containing the job_id and the sample
            num_procs (int): number of processors
            experiment_name (str): name of QUEENS experiment.
            experiment_dir (Path): Path to QUEENS experiment directory.

        Returns:
            Result and potentially the gradient
        """
