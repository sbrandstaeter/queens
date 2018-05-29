import abc

class RegressionApproximation(metaclass=abc.ABCMeta):
    """ Base class for regression approximations

        Regression approxiamtion are regression models/approaches that are called
        regression approximations within QUEENS to avoid the term model.

    """

    @classmethod
    def from_options(cls, approx_options, Xtrain, Ytrain):
        """ Create approxiamtion from options dict

        Args:
            approx_options (dict): Dictionary with approximation options
            Xtrain (np.array):     Training inputs
            Ytrain (np.array):     Training outputs

        Returns:
            regression_approximation: Approximation object

        """
        from .gp_approximation_gpy import GPGPyRegression
        approx_dict = {'gp_approximation_gpy': GPGPyRegression}

        approximation_class = approx_dict[approx_options["type"]]

        return approximation_class.from_options(approx_options, Xtrain, Ytrain)

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def predict(self, Xnew):
        pass
