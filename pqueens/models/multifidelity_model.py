"""Multi-fidelity model class."""

from pqueens.interfaces import from_config_create_interface

from .model import Model
from .simulation_model import SimulationModel


class MultifidelityModel(Model):
    """Multi-fidelity model class.

    The multi-fidelity model class holds a sequence of simulation models for
    multi-fidelity sampling schemes.

    Attributes:
        eval_cost_per_level (list):        Cost for one model evaluation.
        num_levels (int):                  Number of levels, i.e. number of models in
                                           multi-fidelity model.
        __model_sequence (list):             List of models comprising multi-fidelity
                                           model.
        __admissible_response_modes (list):  List with admissible response modes.
        __active_lf_model_ind (int):         Index of current low-fidelity model.
        __active_hf_model_ind (int):         Index of current high-fidelity model.
        response_mode (string):            Current response mode.
    """

    def __init__(self, model_name, model_sequence, eval_cost_per_level):
        """Initialize multi-fidelity model.

        Args:
            model_name (string):        Name of model
            model_sequence (list):      List with SimulationModels
            eval_cost_per_level (list): List with model evaluation cost
        """
        super().__init__(model_name)

        self.eval_cost_per_level = eval_cost_per_level
        self.num_levels = len(model_sequence)
        self.__model_sequence = model_sequence
        self.__admissible_response_modes = ['uncorrected_lofi', 'aggregated_model', 'bypass_lofi']

        self.__active_lf_model_ind = None
        self.__active_hf_model_ind = None
        self.response_mode = None

    @classmethod
    def from_config_create_model(cls, model_name, config):
        """Create multi-fidelity model from problem description.

        Args:
            model_name (string): Name of model
            config (dict):       Dictionary containing problem description

        Returns:
            multifidelity_model: Instance of MultifidelityModel
        """
        # get options
        model_options = config[model_name]
        model_hierarchy = model_options['model_hierarchy']
        parameters = model_options["parameters"]
        model_parameters = config[parameters]
        model_evaluation_cost = model_options['eval_cost_per_level']

        # create submodels
        sub_models = []
        for sub_model_name in model_hierarchy:
            sub_model_options = config[sub_model_name]
            sub_interface_name = sub_model_options["interface_name"]
            if sub_model_options["type"] != 'simulation_model':
                raise ValueError(
                    'Multifidelity models can only have simulation models as sub models'
                )
            # TODO check for same parameters
            sub_interface = from_config_create_interface(sub_interface_name, config)
            sub_models.append(SimulationModel(sub_model_name, sub_interface))

        return cls(model_name, sub_models, model_evaluation_cost)

    def evaluate(self):
        """Evaluate model with current set of variables.

        Returns:
            TODO_doc
        """
        # switch according to response mode
        if self.response_mode == 'uncorrected_lofi':
            self.response = self._lofi_model().interface.evaluate(self.variables)
        elif self.response_mode == 'aggregated_model':
            self.response = self._lofi_model().interface.evaluate(self.variables)
            self.response = [self.response, self._hifi_model().interface.evaluate(self.variables)]
            # this case needs to be adapted to new output structure
            raise NotImplementedError
        elif self.response_mode == 'bypass_lofi':
            self.response = self._hifi_model().interface.evaluate(self.variables)
        else:
            raise RuntimeError("Unknown response type")

        return self.response

    def set_response_mode(self, new_response_mode):
        """Set response mode of multi-fidelity model.

        Args:
            new_response_mode: TODO_doc
        """
        if new_response_mode in self.__admissible_response_modes:
            self.response_mode = new_response_mode
        else:
            raise ValueError("Unsupported response mode")

    def solution_level_cost(self):
        """Return solution cost for each model level.

        Returns:
            TODO_doc
        """
        return self.eval_cost_per_level

    def set_hifi_model_index(self, hifi_index):
        """Choose which model is used as hi-fi model.

        Args:
            hifi_index: TODO_doc
        """
        if hifi_index > len(self.__model_sequence):
            raise ValueError("Index for hi-fi model is out of range")
        else:
            self.__active_hf_model_ind = hifi_index

    def set_lofi_model_index(self, lofi_index):
        """Choose which model is used as lo-fi model.

        Args:
            lofi_index: TODO_doc
        """
        if lofi_index > len(self.__model_sequence):
            raise ValueError("Index for lo-fi model is out of range")
        else:
            self.__active_lf_model_ind = lofi_index

    def _lofi_model(self):
        """Get current low-fidelity model."""
        return self.__model_sequence[self.__active_lf_model_ind]

    def _hifi_model(self):
        """Get current high-fidelity model."""
        return self.__model_sequence[self.__active_hf_model_ind]
