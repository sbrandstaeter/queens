"""Iterator to run a model on predefined input points."""

import logging
import time

import numpy as np

from pqueens.iterators.iterator import Iterator
from pqueens.utils.ascii_art import print_points_iterator
from pqueens.utils.process_outputs import write_results

_logger = logging.getLogger(__name__)


class PointsIterator(Iterator):
    """Iterator at given input points.

    Attributes:
        input_values (np.array): Array with all samples
        write_results (bool): Export data
        output (np.array): Array with all model outputs
    """

    def __init__(self, model, global_settings, points, result_description):
        """Initialise Iterator.

        Args:
            model (obj, optional): Model to be evaluated by iterator
            global_settings (dict, optional): Settings for the QUEENS run
            points (dict): Dictionary with name and samples
            result_description (dict): Settings for storing
        """
        super().__init__(model, global_settings)
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
        lens = [len(d) for d in points]

        if len(set(lens)) != 1:
            message = ", ".join([f"{n}: {l}" for n, l in zip(self.parameters.names, lens)])
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

                write_results(
                    results,
                    self.global_settings["output_dir"],
                    self.global_settings["experiment_name"],
                )
