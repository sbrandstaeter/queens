"""Adjoint model."""

import logging

import numpy as np

import pqueens.database.database as DB_module
from pqueens.interfaces import from_config_create_interface
from pqueens.models.gradient_models.gradient_model import GradientModel
from pqueens.utils.config_directories import current_job_directory
from pqueens.utils.output_writer import write_to_csv

_logger = logging.getLogger(__name__)


class AdjointModel(GradientModel):
    """Adjoint model.

    Attributes:
        adjoint_file_name (str): Name of the adjoint file that contains the evaluated
                                 derivative of the functional w.r.t. to the simulation output.
        gradient_interface (obj): Interface object for the adjoint simulation run.
        db (database_obj): QUEENS database
        experiment_name (str): Name of the current QUEENS experiment
    """

    def __init__(
        self, model_name, forward_model, adjoint_file_name, gradient_interface, db, experiment_name
    ):
        """Initialize model.

        Args:
            model_name (str): Model name
            forward_model (obj): Forward model on which the likelihood model is based
            adjoint_file_name (str): Name of the adjoint file that contains the evaluated
                                     derivative of the functional w.r.t. to the simulation output.
            gradient_interface (obj): Interface object for the adjoint simulation run.
            db (database_obj): QUEENS database
            experiment_name (str): Name of the current QUEENS experiment
        """
        super().__init__(model_name, forward_model)
        self.gradient_interface = gradient_interface
        self.adjoint_file_name = adjoint_file_name
        self.db = db
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
        forward_model = super().get_base_attributes_from_config(model_name, config)
        model_options = config[model_name]
        adjoint_file_name = model_options.get("adjoint_file_name", "adjoint_grad_objective.csv")
        gradient_interface_name = model_options["gradient_interface_name"]
        gradient_interface = from_config_create_interface(gradient_interface_name, config)
        db = DB_module.database
        experiment_name = config["global_settings"]["experiment_name"]
        return cls(
            model_name=model_name,
            forward_model=forward_model,
            adjoint_file_name=adjoint_file_name,
            gradient_interface=gradient_interface,
            db=db,
            experiment_name=experiment_name,
        )

    def evaluate(self, samples, **kwargs):
        """Evaluate forward model with current set of variables."""
        return self.forward_model.evaluate(samples)

    def _grad(self, samples, upstream):
        """Evaluate gradient of model with current set of samples.

        Args:
            samples (np.array): Evaluated samples
            upstream (np.array): Upstream gradient
        """
        forward_model_output = (
            self.forward_model.response
        )  # TODO: write this also into adjoint file

        # get last job_id from data base
        experiment_dir = self.gradient_interface.experiment_dir
        current_batch_number = self.gradient_interface.batch_number + 1
        last_job_batch = self.db.load(self.experiment_name, current_batch_number, "jobs_driver")
        last_job_ids = [job['id'] for job in last_job_batch]

        # write adjoint data for each sample to adjoint files in old job directories
        for job_id, grad_objective in zip(last_job_ids, upstream):
            job_dir = current_job_directory(experiment_dir, job_id)
            adjoint_file_path = job_dir.joinpath(self.adjoint_file_name)
            write_to_csv(adjoint_file_path, np.atleast_2d(grad_objective))

        # evaluate the adjoint model
        gradient_response = self.gradient_interface.evaluate(samples)['mean']

        return gradient_response
