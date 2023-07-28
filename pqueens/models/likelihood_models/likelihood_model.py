"""Module to define likelihood functions."""

import abc

import numpy as np

from pqueens.models.model import Model


class LikelihoodModel(Model):
    """Base class for likelihood models.

    Attributes:
        forward_model (obj): Forward model on which the likelihood model is based
        y_obs (np.array): Observation data
    """

    def __init__(self, forward_model, y_obs):
        """Initialize the likelihood model.

        Args:
            forward_model (obj): Forward model that is evaluated during the likelihood evaluation
            y_obs (array_like): Observation data
        """
        super().__init__()
        self.forward_model = forward_model
        self.y_obs = np.array(y_obs)

    @abc.abstractmethod
    def evaluate(self, samples):
        """Evaluate model with current set of input samples.

        Args:
            samples (np.ndarray): Input samples
        """

    @abc.abstractmethod
    def grad(self, samples, upstream_gradient):
        r"""Evaluate gradient of model w.r.t. current set of input samples.

        Consider current model f(x) with input samples x, and upstream function g(f). The provided
        upstream gradient is :math:`\frac{\partial g}{\partial f}` and the method returns
        :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`.

        Args:
            samples (np.array): Input samples
            upstream_gradient (np.array): Upstream gradient function evaluated at input samples
                                          :math:`\frac{\partial g}{\partial f}`

        Returns:
            gradient (np.array): Gradient w.r.t. current set of input samples
                                 :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`
        """
