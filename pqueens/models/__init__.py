"""Models.

The model package contains several types of models to be used in the
context of UQ. Within QUEENS the a model class of object holds and
stores the input and output data and can evaluate itself to produce
data.
"""
from pathlib import Path

from pqueens.utils.import_utils import get_module_attribute


def from_config_create_model(model_name, config):
    """Create model from problem description.

    Args:
        model_name (string):    Name of model
        config  (dict):         Dictionary with problem description

    Returns:
        model: Instance of model class
    """
    from pqueens.models import likelihood_models

    from .bmfmc_model import BMFMCModel
    from .data_fit_surrogate_model import DataFitSurrogateModel
    from .data_fit_surrogate_model_mf import MFDataFitSurrogateModel
    from .multifidelity_model import MultifidelityModel
    from .simulation_model import SimulationModel

    model_dict = {
        'simulation_model': SimulationModel,
        'datafit_surrogate_model': DataFitSurrogateModel,
        'datafit_surrogate_model_mf': MFDataFitSurrogateModel,
        'multi_fidelity_model': MultifidelityModel,
        'bmfmc_model': BMFMCModel,
        'likelihood_model': likelihood_models,
    }

    model_options = config[model_name]
    if model_options["type"] in model_dict.keys():
        model_class = model_dict[model_options["type"]]
    elif model_options.get("path_external_python_module"):
        module_path = model_options["path_external_python_module"]
        module_attribute = model_options["type"]
        model_class = get_module_attribute(module_path, module_attribute)
    else:
        raise ModuleNotFoundError(
            f"The module '{model_options['type']}' could not be found!\n"
            f"Valid internal modules are: {model_dict.keys()}.\n"
            "If you provided an external Python module, make sure you specified \n"
            "the correct class name under the 'type' keyword!"
        )

    return model_class.from_config_create_model(model_name, config)
