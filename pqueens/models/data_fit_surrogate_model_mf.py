import numpy as np
from pqueens.iterators.iterator import Iterator
from pqueens.interfaces.interface import Interface
from . model import Model

# TODO add test

class MFDataFitSurrogateModel(Model):
    """ Multi-fidelity Surrogate model class

        Attributes:
            interface (interface):              mf approximation interface
            subordinate_model (model):          (thruth) model
            subordinate_iterator (iterator):    Iterator for subordinate model

    """

    def __init__(self, model_name, interface, model_parameters, subordinate_model,
                 subordinate_iterator):
        """ Initialize data fit surrogate model

        Args:
            model_name (string):             Name of model
            interface (interface):           Interface to simulator
            model_parameters (dict):         Dictionary with description of
                                             model parameters
            subordinate_model (model):       Model the surrogate is based on
            subordinate_iterator (Iterator): Iterator to evaluate the subordinate
                                             model with the purpose of getting
                                             training data

        """
        super(MFDataFitSurrogateModel, self).__init__(model_name, model_parameters)
        self.interface = interface
        self.subordinate_model = subordinate_model
        self.subordinate_iterator = subordinate_iterator


    @classmethod
    def from_config_create_model(cls, model_name, config):
        """  Create data fit surrogate model from problem description

        Args:
            model_name (string): Name of model
            config (dict):       Dictionary containing problem description

        Returns:
            data_fit_surrogate_model:   Instance of DataFitSurrogateModel

        """
        # get options
        model_options = config[model_name]
        interface_name = model_options["interface"]
        parameters = model_options["parameters"]
        model_parameters = config[parameters]

        subordinate_model_name = model_options["subordinate_model"]
        subordinate_iterator_name = model_options["subordinate_iterator"]

        # create subordinate model
        subordinate_model = Model.from_config_create_model(subordinate_model_name,
                                                           config)

        # create subordinate iterator
        subordinate_iterator = Iterator.from_config_create_iterator(config,
                                                                    subordinate_iterator_name,
                                                                    subordinate_model)
        # TODO add check if we have a multi-fidelity iterator

        # create interface
        interface = Interface.from_config_create_interface(interface_name, config)

        # TODO check that we have a multi-fidelity interface

        return cls(model_name, interface, model_parameters, subordinate_model,
                   subordinate_iterator)

    def evaluate(self):
        """ Evaluate model with current set of variables

        Returns:
            np.array: Results correspoding to current set of variables
        """
        if not self.interface.is_initiliazed():
            self.build_approximation()

        self.response = self.interface.map(self.variables)
        return np.reshape(np.array(self.response), (-1, 1))

    def build_approximation(self):
        """ Build underlying approximation """

        self.subordinate_iterator.run()

        # get samples and results
        X = self.subordinate_iterator.samples
        Y = self.subordinate_iterator.outputs

        # train regression model on the data
        self.interface.build_approximation(X, Y)


    def compute_error_measures(self, y_act, y_pred, measures):
        """ Compute error measures

            Compute based on difference between predicted and actual values.

            Args:
                y_act (np.array):  Actual values
                y_pred (np.array): Predicted values
                measures (list):   Dictionary with desired error measures

            Returns:
                dict: Dictionary with error measures and correspoding error values
        """
        error_measures = {}
        for measure in measures:
            error_measures[measure] = self.compute_error(y_act, y_pred, measure)
        return error_measures

    # TODO move this to more general place
    def compute_error(self, y_act, y_pred, measure):
        """ Compute error for given a specific error measure

            Args:
                y_act (np.array):  Actual values
                y_pred (np.array): Predicted values
                measure (str):     Desired error metric

            Returns:
                float: error based on desired metric

            Raises:

        """
        # TODO checkout raises field
        if measure == "sum_squared":
            error = np.sum((y_act - y_pred)**2)
        elif measure == "mean_squared":
            error = np.mean((y_act - y_pred)**2)
        elif measure == "root_mean_squared":
            error = np.sqrt(np.mean((y_act - y_pred)**2))
        elif measure == "sum_abs":
            error = np.sum(np.abs(y_act - y_pred))
        elif measure == "mean_abs":
            error = np.mean(np.abs(y_act - y_pred))
        elif measure == "abs_max":
            error = np.max(np.abs(y_act - y_pred))
        else:
            raise NotImplementedError("Desired error measure is unknown!")
        return error
