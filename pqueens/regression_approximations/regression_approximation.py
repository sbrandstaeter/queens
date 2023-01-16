"""Regression approximation base class."""
import abc


class RegressionApproximation(metaclass=abc.ABCMeta):
    """Base class for regression approximations.

    Regression approximation are regression models/approaches, that are
    called regression approximations within QUEENS to avoid the term
    model.
    """

    @abc.abstractmethod
    def train(self):
        """Train the regression approximator."""
        pass

    @abc.abstractmethod
    def predict(self):
        """Evaluate the regression approximator."""
        pass
