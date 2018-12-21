import numpy as np
from pqueens.iterators.iterator import Iterator
from pqueens.interfaces.interface import Interface
from . model import Model
from . simulation_model import SimulationModel
from . joint_prob_model import JointProbModel

class BMFMCModel(Model):
    """ Generalized Bayesian multi-fidelity model class

    The generalized Bayesian multi-fidelity model class holds a high-fideliy
    model, several low fidelity models, the joint random process model for
    the mapping lofi to hifi, models for lofi output distributions as well as the
    posterior models for the output distributions of each lofi-hifi mapping.

    Attributes:
        model_sequence (list):             List of models comprising multi-fidelity
                                           model
        eval_cost_per_level (list):        Cost for one model evaluation
        num_levels (int):                  Number of levels, i.e., number models in
                                           multi-fidelity model
        admissible_response_modes (list):  List with admissible response modes
        response_mode (string):            Current response mode
        active_lf_model_ind (int):         Index of current low-fidelity model
        active_hf_model_ind (int):         Index of current low-fidelity model
        joint_output_model:                Joint probability model for multi-fidelity outputs
    """


    def __init__(self, model_name, model_parameters, model_sequence,
                 eval_cost_per_level, data_iterator):
        """ Initialize BMFMC model

        Args:
            model_name (string):        Name of model
            model_parameters (dict):    Model parameters
            model_sequence (list):      List with SimulationModels (First model
            is the hifi model followed by multiple lofi models
            eval_cost_per_level (list): List with model evaluation cost

        """
        # Initialize base class (Model)
        super(BMFMCModel, self).__init__(model_name, model_parameters)
        # General initializations for BMFMC model
        self.eval_cost_per_level = eval_cost_per_level
       # self.num_levels = len(model_sequence)
        self.__model_sequence = model_sequence # TODO: Check how this is used -> we need it only for the lofis
        self.__active_lf_model_ind = None
        self.data_iterator = data_iterator

    @classmethod
    def from_config_create_model(cls, model_name, config):
        """  Create multi-fidelity model from problem description

        Args:
            model_name (string): Name of model
            config (dict):       Dictionary containing problem description

        Returns:
            multifidelity_model:   Instance of MultifidelityModel

        """
#####get options for generic BMFMC model containing all other models ##########################################################
        model_options = config[model_name]
        model_hierarchy = model_options['model_hierarchy']
        parameters = model_options["parameters"] # random input parameters / fields (in this case the same for all models!)
        model_parameters = config[parameters]
        model_evaluation_cost = model_options['eval_cost_per_level']

######### create the actual models contained in the BMFMC model ##################

        # create the lofi models
        lofi_models = []
        for lofi_name in model_hierarchy[1:]: # fist model is hifi
            lofi_options = config[lofi_name]
            lofi_interface_name = lofi_options["interface"]
            if lofi_options["type"] != 'simulation_model':
                raise ValueError('BMFMC models can currently only have simulation models as Lofi models')
            lofi_interface = Interface.from_config_create_interface(lofi_interface_name, config)
            lofi_models.append(SimulationModel(lofi_name, lofi_interface, model_parameters))

        # create the hifi model (points are equivalent to
        # from_config_create_model)
        hifi_model = []
        hifi_options = config[model_hierarchy[0]]
        hifi_interface_name = hifi_options["interface"]
        if hifi_options["type"] != 'simulation_model':
            raise ValueError('BMFMC hifi model can currently only be a simulation model !')
        hifi_interface = Interface.from_config_create_interface(hifi_interface_name, config)
        hifi_model.append(SimulationModel(model_hierarchy[0], hifi_interface, model_parameters))

        # create the model for the joint probability (currenly only GP possible)
        joint_model_name = model_options['joint_model']
        joint_model_options = config[joint_model_name]
        joint_interface_name = joint_model_options['interface']
        joint_interface = Interface.from_config_create_interface(joint_interface_name, config)
        joint_model = JointProbModel(joint_model_name, joint_interface)
        # or better Model.from_config_create ?

        # create data_iterator
        data_iterator_name = model_options['data_iterator']
        data_iterator_options = config[data_iterator_name]
        data_iterator = Iterator.from_config_create_iterator(config,
                                                             data_iterator_name, )

        return cls(model_name, model_parameters, joint_model_object,
                   model_evaluation_cost, data_iterator)

    def evaluate(self):
        """ Evaluate the distinct models according to the configuration
        settings """

        # Currently we assume that for the learning the model's joint output
        # probability we have to simulate the same random input set
        # realizations on all models. Hence X for the joint distribution is the
        # same on all models.
        # Joint model will be constructured elsewhere!
        self.response = self._lofi_model().interface.map(self.variables) #lofis
        self.response = [self.response,
                         self.__model_sequence[0].interface.map(self.variables)]#hifi
        return self.response

############################################################################
    def solution_level_cost(self):
        """ Return solution cost for each model level """
        return self.eval_cost_per_level

    def _lofi_model(self):
        """ Get current low-fidelity model """
        return self.__model_sequence[self.__active_lf_model_ind]

    def set_lofi_model_index(self, lofi_index):
        """ Choose which model is used as lo-fi model """
        if lofi_index > len(self.__model_sequence):
            raise ValueError("Index for lo-fi model is out of range")
        else:
            self.__active_lf_model_ind = lofi_index

