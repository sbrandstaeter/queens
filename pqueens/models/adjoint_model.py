"""Adjoint model."""

import logging

from pqueens.interfaces import from_config_create_interface
from pqueens.models.model import Model
from pqueens.utils.config_directories import current_job_directory
from pqueens.utils.io_utils import write_to_csv

_logger = logging.getLogger(__name__)


class AdjointModel(Model):
    """Adjoint model.

    Attributes:
        adjoint_file (str): Name of the adjoint file that contains the evaluated derivative of the
                            functional w.r.t. to the simulation output.
        gradient_interface (obj): Interface object for the adjoint simulation run.
        experiment_name (str): Name of the current QUEENS experiment
    """

    def __init__(
        self,
        model_name,
        global_settings,
        interface,
        gradient_interface,
        adjoint_file="adjoint_grad_objective.csv",
    ):
        """Initialize model.

        Args:
            global_settings (dict): Dictionary containing global settings for the QUEENS run.
            model_name (str): Model name
            interface (obj): Interface object for forward simulation run
            gradient_interface (obj): Interface object for the adjoint simulation run.
            adjoint_file (str): Name of the adjoint file that contains the evaluated derivative of
                                the functional w.r.t. to the simulation output.
        """
        super().__init__(model_name)
        self.interface = interface
        self.gradient_interface = gradient_interface
        self.adjoint_file = adjoint_file
        self.experiment_name = global_settings['experiment_name']

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
        global_settings = config["global_settings"]

        model_options = config[model_name]
        interface_name = model_options.pop('interface_name')
        interface = from_config_create_interface(interface_name, config)
        gradient_interface_name = model_options.pop("gradient_interface_name")
        gradient_interface = from_config_create_interface(gradient_interface_name, config)
        model_options.pop('type')

        return cls(
            model_name=model_name,
            interface=interface,
            gradient_interface=gradient_interface,
            global_settings=global_settings,
            **model_options
        )

    def evaluate(self, samples, **kwargs):
        """Evaluate forward model with current set of variables."""
        self.response = self.interface.evaluate(samples)
        return self.response

    def grad(self, samples, upstream):
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
            adjoint_file_path = job_dir.joinpath(self.adjoint_file)
            write_to_csv(adjoint_file_path, grad_objective.reshape(1, -1))

        # evaluate the adjoint model
        gradient_response = self.gradient_interface.evaluate(samples)['mean']

        return gradient_response
