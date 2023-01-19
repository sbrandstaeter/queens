"""Module to define gradient models."""

import abc

from pqueens.models import from_config_create_model
from pqueens.models.model import Model


class GradientModel(Model):
    """Base class for gradient models.

    Attributes:
        forward_model (obj): Forward model
    """

    def __init__(
        self,
        model_name,
        forward_model,
    ):
        """Initialize the gradient model.

        Args:
            model_name (str): Name of the underlying model in input file
            forward_model (obj): Forward model that is evaluated during the likelihood evaluation
        """
        super().__init__(model_name)
        self.forward_model = forward_model

    @staticmethod
    def get_base_attributes_from_config(model_name, config):
        """Get the attributes of the base class from config.

        Args:
            model_name (str): Name of the model in the input file
            config (dict): Config of the QUEENS run

        Returns:
            forward_model (obj): Forward model
        """
        model_options = config[model_name]
        forward_model_name = model_options.get("forward_model_name")
        forward_model = from_config_create_model(forward_model_name, config)
        return forward_model

    @abc.abstractmethod
    def evaluate(self, samples):
        """Evaluate forward model with current set of variables."""
        return self.forward_model.evaluate(samples)
