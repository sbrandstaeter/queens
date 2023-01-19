"""Adjoint model."""

import logging

import numpy as np

from pqueens.models.gradient_models.gradient_model import GradientModel
from pqueens.utils.grad_utils import Tracer

_logger = logging.getLogger(__name__)


class AdjointModel(GradientModel):
    """Adjoint model."""

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
        """Create adjoint model from problem description.

        Args:
            model_name (str): Name of the model
            config (dict): Dictionary containing problem description

        Returns:
            instance of AdjointModel class
        """
        forward_model = super().get_base_attributes_from_config(model_name, config)
        return cls(model_name=model_name, forward_model=forward_model)

    def evaluate_and_grad(self, samples, tracer=Tracer()):
        model_output = self.forward_model.evaluate(samples)
        objective_output, objective_grad_model = tracer.evaluate_and_grad(model_output)
        objective_grad_x = self.solve_adjoint(objective_grad_model)
        return objective_output, objective_grad_x

