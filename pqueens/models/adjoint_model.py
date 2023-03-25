"""Adjoint model."""

import logging

import numpy as np

from pqueens.interfaces import from_config_create_interface
from pqueens.models.model import Model
from pqueens.utils.config_directories import current_job_directory
from pqueens.utils.io_utils import write_to_csv

_logger = logging.getLogger(__name__)


class AdjointModel(Model):
    """Adjoint model.

    Attributes:
        adjoint_file_name (str): Name of the adjoint file that contains the evaluated
                                 derivative of the functional w.r.t. to the simulation output.
        gradient_interface (obj): Interface object for the adjoint simulation run.
        experiment_name (str): Name of the current QUEENS experiment
    """

    def __init__(
        self, model_name, interface, gradient_interface, adjoint_file_name, experiment_name
    ):
        """Initialize model.

        Args:
            model_name (str): Model name
            interface (obj): Interface object for forward simulation run
            gradient_interface (obj): Interface object for the adjoint simulation run.
            adjoint_file_name (str): Name of the adjoint file that contains the evaluated
                                     derivative of the functional w.r.t. to the simulation output.
            experiment_name (str): Name of the current QUEENS experiment
        """
        super().__init__(model_name)
        self.interface = interface
        self.gradient_interface = gradient_interface
        self.adjoint_file_name = adjoint_file_name
        self.experiment_name = experiment_name

    @classmethod
    def from_config_create_model(
        cls,
        model_name,
        config,
    ):
        """Create adjoint model from problem description.

        Args:
            model_name (str): Name of the model
            config (dict): Dictionary containing problem description

        Returns:
            instance of AdjointModel class
        """
        model_options = config[model_name]
        interface_name = model_options['interface_name']
        interface = from_config_create_interface(interface_name, config)
        gradient_interface_name = model_options["gradient_interface_name"]
        gradient_interface = from_config_create_interface(gradient_interface_name, config)

        adjoint_file_name = model_options.get("adjoint_file_name", "adjoint_grad_objective.csv")
        experiment_name = config["global_settings"]["experiment_name"]
        return cls(
            model_name=model_name,
            interface=interface,
            gradient_interface=gradient_interface,
            adjoint_file_name=adjoint_file_name,
            experiment_name=experiment_name,
        )

    def evaluate(self, samples, **kwargs):
        """Evaluate forward model with current set of variables."""
        self.response = self.interface.evaluate(samples)
        return self.response

    def _grad(self, samples, upstream):
        """Evaluate gradient of model with current set of samples.

        Args:
            samples (np.array): Evaluated samples
            upstream (np.array): Upstream gradient
        """
        # get last job_ids
        last_job_ids = self.interface.job_ids[-samples.shape[0] :]

        # write adjoint data for each sample to adjoint files in old job directories
        for job_id, grad_objective in zip(last_job_ids, upstream):
            job_dir = current_job_directory(self.gradient_interface.experiment_dir, job_id)
            adjoint_file_path = job_dir.joinpath(self.adjoint_file_name)
            write_to_csv(adjoint_file_path, np.atleast_2d(grad_objective))

        # evaluate the adjoint model
        gradient_response = self.gradient_interface.evaluate(samples)['mean']

        return gradient_response
