"""Finite difference model."""

import logging

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
        self.response_grad = None

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

    def evaluate(self, samples):
        """Evaluate forward model with current set of variables."""
        finite_difference_star_inputs = self.create_finite_difference_star(samples)
        finite_difference_star_outputs = self.forward_model.evaluate(finite_difference_star_inputs)
        self.response, self.response_grad = self.evaluate_finite_difference_star(
            finite_difference_star_inputs, finite_difference_star_outputs)
        return self.response

    def grad(self, samples, tracer=Tracer()):
        return self.response_grad

    def create_finite_difference_star(self, samples):
        raise NotImplementedError('this has to be implemented')

    def evaluate_finite_difference_star(self, inputs, outputs):
        raise NotImplementedError('this has to be implemented')

