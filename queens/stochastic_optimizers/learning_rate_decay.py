"""Learning rate decay for stochastic optimization."""

import abc
import logging

_logger = logging.getLogger(__name__)


class LearningRateDecay(metaclass=abc.ABCMeta):
    """Base class for learning rate decay."""

    @abc.abstractmethod
    def __call__(self, learning_rate, params, gradient):
        """Adapt learning rate.

        Args:
            learning_rate (float): Current learning rate
            params (np.array): Current parameters
            gradient (np.array): Current gradient

        Returns:
            learning_rate (float): Adapted learning rate
        """


class LogLinearLearningRateDecay(LearningRateDecay):
    """Log linear learning rate decay.

    Attributes:
        slope (float): Logarithmic slope
        iteration (int): Current iteration
    """

    def __init__(self, slope):
        """Initialize LogLinearLearningRateDecay.

        Args:
            slope (float): Logarithmic slope
        """
        self.slope = slope
        self.iteration = 0

    def __call__(self, learning_rate, params, gradient):
        """Adapt learning rate.

        Args:
            learning_rate (float): Current learning rate
            params (np.array): Current parameters
            gradient (np.array): Current gradient

        Returns:
            learning_rate (float): Adapted learning rate
        """
        self.iteration += 1
        learning_rate /= self.iteration**self.slope
        return learning_rate
