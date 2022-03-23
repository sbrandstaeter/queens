"""Class for mapping input variables to responses using an approximation."""
import numpy as np
from sklearn.model_selection import KFold

from pqueens.regression_approximations import from_config_create_regression_approximation

from .interface import Interface


class ApproximationInterface(Interface):
    """Class for mapping input variables to responses using an approximation.

        The ApproximationInterface uses a so-called regression approximation,
        which just another name for a regression model that is used in this context
        to avoid confusion and not having to call everything a model.

        For now this interface holds only one approximation object. In the future,
        this could be extended to multiple objects

    Attributes:
        name (string):                 Name of interface
        variables (dict):              Dictionary with variables
        config (dict):   Problem description (input file)
        approximation (regression_approximation):   Approximation object
        approximation_init (bool):            Flag whether or not approximation has been
                                       initialized
    """

    def __init__(self, interface_name, config, approximation_name, variables):
        """Create interface.

        Args:
            interface_name (string):     Name of interface
            config (dict): Problem description (input file)
            approximation_name (str): Name of approximation model in config
            variables (dict):  Dictionary with variables
        """
        self.name = interface_name
        self.variables = variables
        self.config = config
        self.approximation_name = approximation_name
        self.approximation = None
        self.approximation_init = False

    @classmethod
    def from_config_create_interface(cls, interface_name, config, driver_name):
        """Create interface from config dictionary.

        Args:
            interface_name (str):   Name of interface
            config (dict):          Dictionary containing problem description
            driver_name (str): Name of the driver that uses this interface
                               (not used here; only for compatability reasons)

        Returns:
            interface:              Instance of ApproximationInterface
        """
        interface_options = config[interface_name]
        approximation_name = interface_options["approximation"]
        config = config
        parameters = config['parameters']

        # initialize object
        return cls(interface_name, config, approximation_name, parameters)

    def evaluate(self, samples):
        """Call the regression approximation prediction.

        Args:
            samples (list):         list of variables objects

        Returns:
            dict:               Dict with results corresponding to samples
        """
        if not self.approximation_init:
            raise RuntimeError("Approximation has not been properly initialized, cannot continue!")

        inputs = []
        for variables in samples:
            params = variables.get_active_variables()
            inputs.append(list(params.values()))

        # get inputs as array and reshape
        num_active_vars = samples[0].get_number_of_active_variables()
        inputs = np.reshape(np.array(inputs), (-1, num_active_vars), order='F')
        output = self.approximation.predict(inputs)
        return output

    def build_approximation(self, x_train, y_train):
        """Build and train underlying regression model.

        Args:
            x_train (np.array):  Training inputs
            y_train (np.array):  Training outputs
        """
        self.approximation = from_config_create_regression_approximation(
            self.config, self.approximation_name, x_train, y_train
        )
        self.approximation.train()
        self.approximation_init = True

    def is_initialized(self):
        """Is the approximation properly initialized."""
        return self.approximation_init

    def cross_validate(self, x_train, y_train, folds):
        """Cross validation function which calls the regression approximation.

        Args:
            x_train (np.array):   Array of inputs
            y_train (np.array):   Array of outputs
            folds (int):    In how many subsets do we split for cv

        Returns:
            np.array:        Array with predictions
        """
        # init output array
        outputs = np.zeros_like(y_train, dtype=float)
        # set random_state=None, shuffle=False)
        # TODO check out randomness feature
        kf = KFold(n_splits=folds)
        kf.get_n_splits(x_train)

        for train_index, test_index in kf.split(x_train):
            # TODO configuration here is not nice
            approximation = from_config_create_regression_approximation(
                self.config, self.approximation_name, x_train[train_index], y_train[train_index]
            )
            approximation.train()
            outputs[test_index] = approximation.predict(x_train[test_index].T, support='f')['mean']

        return outputs
