import numpy as np
from sklearn.model_selection import KFold
from pqueens.regression_approximations.regression_approximation import RegressionApproximation
from .interface import Interface

class BmfmcInterface(Interface):
    """ Class for grouping output of several simulators with identical input to
        one python touple. This is basically a version of the
        approximation_interface class that allows for vectorized mapping and
        implicit function relationships.

    Attributes:
        name (string):                  name of interface
        variables (dict):               dictionary with variables
        function (function object):     adress of database to use

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


# TODO: Samples are chosen by iterator but can now be python tuples for
# implicit functions rather than plain x_vec data (For now it stays a x--> y
# mapping for the first trials)
    def map(self, samples):
        """ Mapping function which calls the regression approximation
        Prediction with the trained regression function / surrogate

        Args:
            samples (list):         list of high and low fidelity models

        Returns:
            dict:               Dict with results correspoding to samples
        """
        if not self.approx_init:
            raise RuntimeError("Approximation has not been properly initialized, cannot continue!")
        inputs = []
        for variables in samples:
            params = variables.get_active_variables()
            inputs.append(list(params.values()))

        # get inputs as array and reshape
        num_active_vars = samples[0].get_number_of_active_variables()
        inputs = np.reshape(np.array(inputs), (-1, num_active_vars), order='F')
        output = self.approximation.predict_f(inputs)
        # TODO: output was changed to contain now mean and variance of
        # predictive distribution. For more multi-fidelity models those values
        # become vectors/ for implicit curves the model should be changed to
        # return the conditional (gaussian) distributions / for now only y mean
        # and variance are predicted as still explicit formulation
        return output

# This function is called in the model class and Xtrain/Ytrain are there called
# via iterator.samples and iterator.db --> Define Points for regression in
# iterator
    def build_approximation(self, Xtrain, Ytrain):
        """ Build and train underlying regression model

        Args:
            Xtrain (np.array):  Training inputs
            Ytrain (np.array):  Training outputs
        """
        # TODO: Check if this syntax still holds for more dimensions or implicit
        # unsupervised learning!
        self.approximation = RegressionApproximation.from_options(self.approximation_config, Xtrain, Ytrain)
        self.approximation.train()
        self.approx_init = True

    def is_initiliazed(self):
        """ Is the approximation properly initialzed """
        return self.approx_init

######## TODO: Check if below is really needed ##################
    def cross_validate(self, X, Y, folds):
        """ Cross validation function which calls the regression approximation

        Args:
            X (np.array):   Array of inputs
            Y (np.array):   Array of outputs
            folds (int):    In how many subsets do we split for cv

        Returns:
            np.array:        Array with predictions
        """
        # init output array
        outputs = np.zeros_like(Y, dtype=float)
        # set random_state=None, shuffle=False)
        # TODO check out random ness feature
        kf = KFold(n_splits=folds)
        kf.get_n_splits(X)

        for train_index, test_index in kf.split(X):
            approximation = RegressionApproximation.from_options(self.approximation_config,
                                                                 X[train_index],
                                                                 Y[train_index])
            approximation.train()
            outputs[test_index], _ = approximation.predict_f(X[test_index])

        return outputs
