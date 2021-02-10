import abc


class RegressionApproximation(metaclass=abc.ABCMeta):
    """ Base class for regression approximations

        Regression approximation are regression models/approaches that are called
        regression approximations within QUEENS to avoid the term model.

    """

    @classmethod
    def from_config_create(cls, config, approx_name, Xtrain, Ytrain):
        """ Create approximation from options dict

        Args:
            config (dict): Dictionary with problem description
            approx_name (str): Name of the approximation model
            Xtrain (npq.array):     Training inputs
            Ytrain (np.array):     Training outputs

        Returns:
            regression_approximation (obj): Approximation object

        """
        from .gp_approximation_gpy import GPGPyRegression
        from .heteroskedastic_GPflow import HeteroskedasticGP
        from .gp_approximation_gpflow import GPFlowRegression
        from .bayesian_neural_network import GaussianBayesianNeuralNetwork

        approx_dict = {
            'gp_approximation_gpy': GPGPyRegression,
            'heteroskedastic_gp': HeteroskedasticGP,
            'gp_approximation_gpflow': GPFlowRegression,
            'gaussian_bayesian_neural_network': GaussianBayesianNeuralNetwork
        }
        approx_options = config[approx_name]
        approximation_class = approx_dict[approx_options["type"]]

        return approximation_class.from_config_create(config, approx_name, Xtrain, Ytrain)

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def predict(self, Xnew):
        pass
