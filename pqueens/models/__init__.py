"""Models.

The model package contains several types of models to be used in the
context of UQ. Within QUEENS, the model class of object holds and stores
the input and output data, and can evaluate itself to produce data.
"""
from pqueens.utils.import_utils import get_module_class

VALID_TYPES = {
    'simulation_model': ['pqueens.models.simulation_model', 'SimulationModel'],
    'datafit_surrogate_model': ['pqueens.models.data_fit_surrogate_model', 'DataFitSurrogateModel'],
    'datafit_surrogate_model_mf': [
        'pqueens.models.data_fit_surrogate_model_mf',
        'MFDataFitSurrogateModel',
    ],
    'multi_fidelity_model': ['pqueens.models.multifidelity_model', 'MultifidelityModel'],
    'bmfmc_model': ['pqueens.models.bmfmc_model', 'BMFMCModel'],
    'gaussian': ['pqueens.models.likelihood_models.gaussian_likelihood', 'GaussianLikelihood'],
    'bmf_gaussian': [
        'pqueens.models.likelihood_models.bayesian_mf_gaussian_likelihood',
        'BMFGaussianModel',
    ],
    'differentiable_simulation_model_fd': [
        'pqueens.models.differentiable_simulation_model_fd',
        'DifferentiableSimulationModelFD',
    ],
    'differentiable_simulation_model_adjoint': [
        'pqueens.models.differentiable_simulation_model_adjoint',
        'DifferentiableSimulationModelAdjoint',
    ],
    'gp_approximation_gpy': [
        'pqueens.models.surrogate_models.gp_approximation_gpy',
        'GPGPyRegressionModel',
    ],
    'heteroskedastic_gp': [
        'pqueens.models.surrogate_models.heteroskedastic_GPflow',
        'HeteroskedasticGPModel',
    ],
    'gp_approximation_gpflow': [
        'pqueens.models.surrogate_models.gp_approximation_gpflow',
        'GPFlowRegressionModel',
    ],
    'gaussian_bayesian_neural_network': [
        'pqueens.models.surrogate_models.bayesian_neural_network',
        'GaussianBayesianNeuralNetworkModel',
    ],
    'gp_jitted': [
        'pqueens.models.surrogate_models.gp_approximation_jitted',
        'GPJittedModel',
    ],
    'gp_approximation_gpflow_svgp': [
        'pqueens.models.surrogate_models.gp_approximation_gpflow_svgp',
        'GPflowSVGPModel',
    ],
    'gaussian_nn': [
        'pqueens.models.surrogate_models.gaussian_neural_network',
        'GaussianNeuralNetworkModel',
    ],
}


def from_config_create_model(model_name, config):
    """Create model from problem description.

    Args:
        model_name (string):    Name of model
        config  (dict):         Dictionary with problem description

    Returns:
        model: Instance of model class
    """
    model_options = config[model_name]
    model_class = get_module_class(model_options, VALID_TYPES)
    model = model_class.from_config_create_model(model_name, config)
    return model
