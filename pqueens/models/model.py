import abc
from copy import deepcopy

import numpy as np

from pqueens.variables.variables import Variables


class Model(metaclass=abc.ABCMeta):
    """ Base class of model hierarchy

         The model hierarchy contains a set of variables, an interface,
        and a set of responses. An iterator operates on the model to map
        the variables into responses using the interface.

        As with the Iterator hierarchy, the purpose of the this base class is
        twofold. One, it defines a unified interface for all derived classes.
        Two, it acts as a factory for the instantiation of model objects.

    Attributes:
        uncertain_parameters (dict):    Dictionary with description of uncertain
                                        parameters
        variables (list):               Set of model variables where model is evaluated
        responses (list):               Set of responses corresponding to variables
    """

    def __init__(self, name, uncertain_parameters):
        """ Init model object

        Args:
            name (string):                  Name of model
            uncertain_parameters (dict):    Dictionary with description of uncertain
                                            parameters
        """
        self.name = name
        self.uncertain_parameters = uncertain_parameters
        self.variables = [Variables.from_uncertain_parameters_create(uncertain_parameters)]
        self.response = [None]

    @classmethod
    def from_config_create_model(cls, model_name, config):
        """ Create model from problem description

        Args:
            model_name (string):    Name of model
            config  (dict):         Dictionary with problem description

        Returns:
            model: Instance of model class

        """
        from .simulation_model import SimulationModel
        from .data_fit_surrogate_model import DataFitSurrogateModel
        from .data_fit_surrogate_model_mf import MFDataFitSurrogateModel
        from .multifidelity_model import MultifidelityModel

        model_dict = {'simulation_model': SimulationModel,
                      'datafit_surrogate_model': DataFitSurrogateModel,
                      'datafit_surrogate_model_mf': MFDataFitSurrogateModel,
                      'multi_fidelity_model' : MultifidelityModel}

        model_options = config[model_name]
        model_class = model_dict[model_options["type"]]
        return model_class.from_config_create_model(model_name, config)


    @abc.abstractmethod
    def evaluate(self):
        """ Evaluate model with current set of variables """
        pass

    def get_parameter(self):
        """ Get complete parameter dictionary

        Return:
            dict: Dictionary with all parameters

        """
        return self.uncertain_parameters

    def update_model_from_sample(self, data_vector):
        """ Update model variables

        Args:
            data_vector (np.array): Vector with variable values

        """
        if len(self.variables) != 1:
            self.variables = deepcopy([self.variables[0]])

        self.variables[0].update_variables_from_vector(data_vector)

    def update_model_from_sample_batch(self, data):
        """ Update model variables

        Args:
            data (np.array): 2d array with variable values

        """
        #temp_variable = deepcopy(self.variables[0])
        #self.variables = []
        #for i in range(data.shape[0]):
        #    data_vector = data[i, :]
        #    temp_variable.update_variables_from_vector(data_vector)
        #    new_var = deepcopy(temp_variable)
        #    self.variables.append(new_var)

        num_variables = len(self.variables)
        num_data_vectors = data.shape[0]

        if num_variables > num_data_vectors:
            del self.variables[num_data_vectors:]

        elif num_variables < num_data_vectors:
            for i in range(num_variables, num_data_vectors):
                self.variables.append(Variables.from_uncertain_parameters_create(self.uncertain_parameters))

        for i in range(num_data_vectors):
            self.variables[i].update_variables_from_vector(data[i, :])
    def convert_array_to_model_variables(self, data):
        """ Convert input data to model variables

            Args:
                data (np.array): 2d array with variable values

            Returns:
                queens.variables: Converted array

            Raises:
        """
        temp = deepcopy(self.variables[0])
        variables = []
        for i in range(data.shape[0]):
            data_vector = data[i, :]
            temp.update_variables_from_vector(data_vector)
            new_var = deepcopy(temp)
            variables.append(new_var)

        return variables

    def check_for_precalculated_response_of_sample_batch(self, sample_batch):
        """
        Check if the batch of samples has already been evaluated.

        Args:
        sample_batch (2d numpy.ndarray): each row corresponds to a sample

        Returns:
        precalculated (bool): True: batch of samples was already calculated

        """

        precalculated = True
        if sample_batch.shape[0] is not len(self.variables):
            # Dimension mismatch:
            # There should be as many samples to check as variables already set.
            precalculated = False
        else:
            for i in range(sample_batch.shape[0]):
                sample_vector = sample_batch[i, :]
                cur_sample_vector = self.variables[i].get_active_variables_vector()
                cur_sample_vector = np.ravel(cur_sample_vector)
                if not np.array_equal(sample_vector, cur_sample_vector):
                    # Sample batch was NOT found to be precalculated.
                    precalculated = False
                    break

        return precalculated
