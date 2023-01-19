"""Finite difference model."""

import logging

import numpy as np

from pqueens.models.gradient_models.gradient_model import GradientModel
from pqueens.utils.grad_utils import Tracer

_logger = logging.getLogger(__name__)


class FiniteDifferenceModel(GradientModel):
    """Finite difference model."""

    def __init__(
        self,
        model_name,
        forward_model,
    ):
        """Initialize model.

        Args:
            model_name (str): Model name
            forward_model (obj): Forward model on which the likelihood model is based
        """
        super().__init__(model_name, forward_model)

    @classmethod
    def from_config_create_model(
        cls,
        model_name,
        config,
    ):
        """Create Finite difference model from problem description.

        Args:
            model_name (str): Name of the model
            config (dict): Dictionary containing problem description

        Returns:
            instance of FiniteDifferenceModel class
        """
        forward_model = super().get_base_attributes_from_config(model_name, config)
        return cls(model_name=model_name, forward_model=forward_model)

    def evaluate_and_grad(self, samples, tracer=Tracer()):
        samples_fd = self.create_finite_difference_star(samples)
        model_output_fd = self.forward_model.evaluate(samples_fd)
        if self.fd_on_model == True:
            model_output, model_grad_x = self.get_output_and_gradient_from_fd(
                model_output_fd, samples_fd)
            objective_output, objective_grad_model = tracer.evaluate_and_grad(model_output)
            objective_grad_x = np.dot(objective_grad_model, model_grad_x)
        else:   # fd on objective
            objective_output_fd = tracer.evaluate(model_output_fd)
            objective_output, objective_grad_x = self.get_output_and_gradient_from_fd(
                objective_output_fd, samples_fd)
        return objective_output, objective_grad_x

