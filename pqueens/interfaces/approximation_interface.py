import numpy as np
from pqueens.regression_approximations.regression_approximation import RegressionApproximation
from .interface import Interface

class ApproximationInterface(Interface):
    """ Class for mapping input variables to responses using an approximation

        The ApproximationInterface uses a so-called regression approximation,
        which just another name for a regression model that is used in this context
        to avoid confusion and not having to call everthing a model.

        For now this interface holds only one approximation object. In the future,
        this could be extendend to multiple objects

    Attributes:
        name (string):                 Name of interface
        variables (dict):              Dictionary with variables
        approximation_config (dict):   Config options for approximation
        approximation (regression_approximation):   Approximation object
        approx_init (bool):            Flag wether or not approximation has been
                                       initialized
    """

    def __init__(self, interface_name, approximation_config, variables):
        """ Create interface

        Args:
            interface_name (string):     Name of interface
            approximation_config (dict): Config options for approximation
            variables (dict):            Dictionary with variables

        """
        self.name = interface_name
        self.variables = variables
        self.approximation_config = approximation_config
        self.approximation = None
        self.approx_init = False

    @classmethod
    def from_config_create_interface(cls, interface_name, config):
        """ Create interface from config dictionary

        Args:
            interface_name (str):   Name of interface
            config (dict):          Dictionary containing problem description

        Returns:
            interface:              Instance of ApproximationInterface
        """
        interface_options = config[interface_name]
        approximation_name = interface_options["approximation"]
        approximation_config = config[approximation_name]
        parameters = config['parameters']

        # initialize object
        return cls(interface_name, approximation_config, parameters)

    def map(self, samples):
        """ Mapping function which calls the regression approximation

        Args:
            samples (list):         list of variables objects

        Returns:
            np.array:               Array with results correspoding to samples
        """
        if not self.approx_init:
            raise RuntimeError("Approximation has not been properly initialzed, cannot continue!")

        inputs = []
        for variables in samples:
            params = variables.get_active_variables()
            inputs.append(list(params.values()))

        # get inputs as array and reshape
        num_active_vars = samples[0].get_number_of_active_variables()
        inputs = np.reshape(np.array(inputs), (-1, num_active_vars), order='F')
        # predict_f reaturns mean and variance, for now return only mean
        outputs, _ = self.approximation.predict_f(inputs)
        return outputs

    def build_approximation(self, Xtrain, Ytrain):
        """ Build and train underlying regression model

        Args:
            Xtrain (np.array):  Training inputs
            Ytrain (np.array):  Training outputs
        """
        self.approximation = RegressionApproximation.from_options(self.approximation_config,
                                                                  Xtrain, Ytrain)
        self.approximation.train()
        self.approx_init = True

    def is_initiliazed(self):
        """ Is the approximation properly initialzed """
        return self.approx_init
