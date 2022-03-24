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
    from .bmfmc_model import BMFMCModel
    from .data_fit_surrogate_model import DataFitSurrogateModel
    from .data_fit_surrogate_model_mf import MFDataFitSurrogateModel
    from .likelihood_models.likelihood_model import LikelihoodModel
    from .multifidelity_model import MultifidelityModel
    from .simulation_model import SimulationModel

    model_dict = {
        'simulation_model': SimulationModel,
        'datafit_surrogate_model': DataFitSurrogateModel,
        'datafit_surrogate_model_mf': MFDataFitSurrogateModel,
        'multi_fidelity_model': MultifidelityModel,
        'bmfmc_model': BMFMCModel,
        'likelihood_model': LikelihoodModel,
    }

    model_options = config[model_name]
    model_class = model_dict[model_options["type"]]
    return model_class.from_config_create_model(model_name, config)
