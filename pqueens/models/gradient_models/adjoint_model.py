"""Adjoint model."""

import logging

from pqueens.models.gradient_models.gradient_model import GradientModel

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

    def evaluate(self, samples):
        """Evaluate forward model with current set of variables."""
        return self.forward_model.evaluate(samples)

    def grad(self, samples, upstream):
        objective_grad = self.solve_adjoint(upstream)
        return objective_grad

    def solve_adjoint(self, grad_objective):
        forward_model_output = self.forward_model.response
        raise NotImplementedError('this has to be implemented')

