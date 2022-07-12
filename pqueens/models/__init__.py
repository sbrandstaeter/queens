"""Models.

The model package contains several types of models to be used in the
context of UQ. Within QUEENS the a model class of object holds and
stores the input and output data and can evaluate itself to produce
data.
"""


def from_config_create_model(model_name, config):
    """Create model from problem description.

    Args:
        model_name (string):    Name of model
        config  (dict):         Dictionary with problem description

    Returns:
        model: Instance of model class
    """
    from pqueens.models import likelihood_models
    from pqueens.utils.import_utils import get_module_attribute
    from pqueens.utils.valid_options_utils import get_option

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
    # additional check to avoid external modules if subtype is active
    # overwriting needs to be done in the subtype otherwise
    if model_options.get("external_python_module") and not model_options.get("subtype"):
        module_path = model_options["external_python_module"]
        module_attribute = model_options.get("type")
        model_class = get_module_attribute(module_path, module_attribute)
    else:
        model_class = get_option(model_dict, model_options.get("type"))

    return model_class.from_config_create_model(model_name, config)
