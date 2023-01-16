"""TODO_doc."""

import abc


class RegressionApproximationMF(metaclass=abc.ABCMeta):
    """Base class for multi-fidelity regression approximations.

    Regression approximation are regression models/approaches, that are
    called regression approximations within QUEENS to avoid the term
    model.
    """

    @abc.abstractmethod
    def train(self):
        """TODO_doc."""
        pass

    @abc.abstractmethod
    def predict(self, Xnew):
        """TODO_doc.

        Args:
            Xnew: TODO_doc
        """
        pass

    @abc.abstractmethod
    def predict_f(self, Xnew):
        """TODO_doc.

        Args:
            Xnew: TODO_doc
        """
        pass

    @abc.abstractmethod
    def predict_f_samples(self, Xnew, num_samples):
        """TODO_doc.

        Args:
            Xnew: TODO_doc
            num_samples: TODO_doc
        """
        pass
