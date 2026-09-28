#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Optimization toolbox."""

import logging
from abc import ABC, abstractmethod

import numpy as np

from queens.global_settings import GlobalSettings
from queens.iterators._iterator import Iterator
from queens.models._model import Model
from queens.parameters import Parameters
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import write_results

_logger = logging.getLogger(__name__)


class Optimization(Iterator, ABC):
    """Base class for optimization iterators.

    This class defines the interface that optimization iterators must implement and provides
    shared functionality for evaluating the model. Model responses are cached to avoid repeated
    evaluations at previously visited positions.

    Attributes:
        result_description (dict): Description of the requested post-processing.
        precalculated_positions (dict): Previously evaluated parameter positions and the
                                        corresponding model responses.
    """

    @log_init_args
    def __init__(
        self,
        model: Model,
        parameters: Parameters,
        global_settings: GlobalSettings,
        result_description: dict,
    ) -> None:
        """Initialize the optimization iterator.

        Args:
            model (Model): Model evaluated during the optimization.
            parameters (Parameters): Definition of the model parameters.
            global_settings (GlobalSettings): Settings of the QUEENS experiment, including its
                                              name and output directory.
            result_description (dict): Description of the requested post-processing.
        """
        super().__init__(model, parameters, global_settings)
        self.result_description = result_description
        self.precalculated_positions = {"position": [], "objective": []}
        self.solution = None

    @abstractmethod
    def objective(self, x0):
        """Evaluate the objective function at a parameter position.

        Args:
            x0 (np.ndarray): Parameter position at which the objective is evaluated.

        Returns:
            float or np.ndarray: Objective value at *x0*.
        """

    def post_run(self) -> None:
        """Analyze the resulting optimum."""
        _logger.info("The optimum:\n\t%s", self.solution.x)

        if self.result_description:
            if self.result_description["write_results"]:
                write_results(
                    self.solution,
                    self.global_settings.result_file(".pickle"),
                )

    def eval_model(self, positions):
        """Evaluate model at defined positions.

        Args:
            positions (np.ndarray): Positions at which the model is evaluated

        Returns:
            f_batch (np.ndarray): Model response
        """
        positions = positions.reshape(-1, self.parameters.num_parameters)
        f_batch = [None] * len(positions)
        new_positions_to_evaluate = []
        new_positions_batch_id = []
        for i, position in enumerate(positions):
            precalculated_output = self.check_precalculated(position)
            if precalculated_output is None:
                new_positions_to_evaluate.append(position)
                new_positions_batch_id.append(i)
            else:
                f_batch[i] = precalculated_output
        if new_positions_to_evaluate:
            new_positions_to_evaluate = np.array(new_positions_to_evaluate)
            f_new = self.model.evaluate(new_positions_to_evaluate)["result"]
            for position_id, output in zip(new_positions_batch_id, f_new):
                f_batch[position_id] = output
            self.precalculated_positions["position"].extend(new_positions_to_evaluate)
            self.precalculated_positions["objective"].extend(f_new)
        f_batch = np.array(f_batch).squeeze()
        return f_batch

    def check_precalculated(self, position):
        """Check if the model was already evaluated at defined position.

        Args:
            position (np.ndarray): Position at which the model should be evaluated

        Returns:
            np.ndarray: Precalculated model response or *None*
        """
        for i, precalculated_position in enumerate(self.precalculated_positions["position"]):
            if np.equal(position, precalculated_position).all():
                return self.precalculated_positions["objective"][i]
        return None
