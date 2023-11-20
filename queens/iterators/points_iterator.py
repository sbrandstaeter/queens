"""Iterator to run a model on predefined input points."""

import logging
import time

import numpy as np

from queens.iterators.iterator import Iterator
from queens.utils.ascii_art import print_points_iterator
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import write_results

_logger = logging.getLogger(__name__)


class PointsIterator(Iterator):
    """Iterator at given input points.

    Attributes:
        result_description (dict): Settings for storing
        output (np.array): Array with all model outputs
        points (dict): Dictionary with name and samples
        points_array (np.ndarray): Array with all samples
    """

    @log_init_args
    def __init__(self, model, parameters, points, result_description):
        """Initialise Iterator.

        Args:
            model (obj, optional): Model to be evaluated by iterator
            parameters (obj): Parameters object
            points (dict): Dictionary with name and samples
            result_description (dict): Settings for storing
        """
        super().__init__(model, parameters)
        self.points = points
        self.result_description = result_description
        self.output = None
        self.points_array = None

    def pre_run(self):
        """Prerun."""
        print_points_iterator()
        _logger.info("Starting points iterator.")

        points = []
        for name in self.parameters.names:
            points.append(np.array(self.points[name]))

        # number of points for each parameter
        points_lengths = [len(d) for d in points]

        # check if the provided number of points is equal for each parameter
        if len(set(points_lengths)) != 1:
            message = ", ".join(
                [f"{n}: {l}" for n, l in zip(self.parameters.names, points_lengths)]
            )
            raise ValueError(
                "Non-matching number of points for the different parameters: " + message
            )
        self.points_array = np.array(points).T

        _logger.info("Number of model calls: %d", len(self.points_array))

    def core_run(self):
        """Run model."""
        start_time = time.time()
        self.output = self.model.evaluate(self.points_array)
        end_time = time.time()
        _logger.info("Model runs done, took %f seconds", end_time - start_time)

    def post_run(self):
        """Write results."""
        if self.result_description is not None:
            if self.result_description.get("write_results"):
                results = {
                    "n_points": len(self.points_array),
                    "points": self.points,
                    "output": self.output,
                }

                write_results(results, self.output_dir, self.experiment_name)
