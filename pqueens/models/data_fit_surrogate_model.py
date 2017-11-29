import numpy as np
from pqueens.iterators.iterator import Iterator
from pqueens.interfaces.interface import Interface
from . model import Model

class DataFitSurrogateModel(Model):
    """ Surrogate model class """

    def __init__(self, model_name, interface, model_parameters, subordinate_model,
                 subordinate_iterator):
        """ Initialize data fit surrogate model

        Args:
            model_name (string):        Name of model
            interface (interface):      Interface to simulator
            model_parameters (dict):    Dictionary with description of
                                        model parameters
            subordinate_model (model):  Model the surrogate is based on
            subordinate_iterator (Iterator): Iterator to evaluate the subordinate
                                             model with the purpose of getting
                                             training data

        """
        super(DataFitSurrogateModel, self).__init__(model_name, interface,
                                                    model_parameters)
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

        # create interface
        interface = Interface.from_config_create_interface(interface_name, config)

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

    def eval_surrogate_accuracy(self):
        # TODO implement this
        raise NotImplementedError
