"""Models.

The model package contains several types of models to be used in the
context of UQ. Within QUEENS the a model class of object holds and
stores the input and output data and can evaluate itself to produce
data.
"""
from pqueens.utils.import_utils import get_module_class


def from_config_create_model(model_name, config):
    """Create model from problem description.

    Args:
        model_name (string):    Name of model
        config  (dict):         Dictionary with problem description

    Returns:
        model: Instance of model class
    """
    valid_types = {
        'simulation_model': ['.simulation_model', 'SimulationModel'],
        'datafit_surrogate_model': ['.data_fit_surrogate_model', 'DataFitSurrogateModel'],
        'datafit_surrogate_model_mf': ['.data_fit_surrogate_model_mf', 'MFDataFitSurrogateModel'],
        'multi_fidelity_model': ['.multifidelity_model', 'MultifidelityModel'],
        'bmfmc_model': ['.bmfmc_model', 'BMFMCModel'],
        'gaussian': ['.likelihood_models.gaussian_likelihood', 'GaussianLikelihood'],
        'bmf_gaussian': ['.likelihood_models.bayesian_mf_gaussian_likelihood', 'BMFGaussianModel'],
    }

    model_options = config[model_name]
    model_type = model_options.get("type")
    model_class = get_module_class(model_options, valid_types, model_type)
    model = model_class.from_config_create_model(model_name, config)

    return model
