import numpy as np
from sklearn.model_selection import KFold
from pqueens.regression_approximations.regression_approximation import RegressionApproximation
from .interface import Interface
import pdb


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

    def __init__(self, approximation_config, variables=None):
        """ Create interface

        Args:
            interface_name (string):     Name of interface
            approximation_config (dict): Config options for approximation
            variables (dict):            Dictionary with variables

        """
        self.variables = variables  # TODO: This is acutally not used I think!
        self.approximation_config = approximation_config
        self.approximation = None
        self.approx_init = False

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
        # inputs = []
        # for variables in samples:
        #   params = variables.get_active_variables()
        #   inputs.append(list(params.values()))

        # get inputs as array and reshape
        # num_active_vars = samples[0].get_number_of_active_variables()
        # inputs = np.reshape(np.array(inputs), (-1, num_active_vars), order='F')
        mean = self.approximation.predict_y(samples.T)['mean']
        # we chose an option with the additional likelihood noise already added to
        # the predictive variance (integration over all possible models already done)
        var = self.approximation.predict_y(samples.T)['variance']
        return mean, var

    def build_approximation(self, Xtrain, Ytrain):
        """ Build and train underlying regression model

        Args:
            Xtrain (np.array):  Training inputs
            Ytrain (np.array):  Training outputs
        """
        self.approximation = RegressionApproximation.from_options(
            self.approximation_config, Xtrain, Ytrain
        )

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
            approximation = RegressionApproximation.from_options(
                self.approximation_config, X[train_index], Y[train_index]
            )
            approximation.train()
            outputs[test_index], _ = approximation.predict_f(X[test_index])

        return outputs
