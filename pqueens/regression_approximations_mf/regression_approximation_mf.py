import abc


class RegressionApproximationMF(metaclass=abc.ABCMeta):
    """Base class for multi-fidelity regression approximations.

    Regression approxiamtion are regression models/approaches that are
    called regression approximations within QUEENS to avoid the term
    model.
    """

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def predict(self, Xnew):
        pass

    @abc.abstractmethod
    def predict_f(self, Xnew):
        pass

    @abc.abstractmethod
    def predict_f_samples(self, Xnew, num_samples):
        pass
