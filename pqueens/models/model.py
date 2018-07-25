import abc
from pqueens.variables.variables import Variables
from copy import deepcopy

class Model(metaclass=abc.ABCMeta):
    """ Base class of model hierarchy

        The model hierarchy contains a set of variables, an interface,
        and a set of responses. An iterator operates on the model to map
        the variables into responses using the interface.

        As with the Iterator hierarchy, the purpose of the this base class is
        twofold. One, it defines a unified interface for all derived classes.
        Two, it acts as a factory for the instanciation of model objects.

    Attributes:
        uncertain_parameters (dict):    Dictionary with description of uncertain
                                        pararameters
        variables (list):               Set of model variables where model is evaluated
        responses (list):               Set of responses corresponding to variables
    """

    def __init__(self, name, uncertain_parameters):
        """ Init model onject

        Args:
            name (string):                  Name of model
            uncertain_parameters (dict):    Dictionary with description of uncertain
                                            pararameters
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
            dict: Dictionary with all pararameters

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
        temp = self.variables[0]
        temp = deepcopy(self.variables[0])
        self.variables = []
        for i in range(data.shape[0]):
            data_vector = data[i, :]
            temp.update_variables_from_vector(data_vector)
            new_var = deepcopy(temp)
            self.variables.append(new_var)

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
