"""Models.

The model package contains several types of models to be used in the
context of UQ. Within QUEENS, the model class of object holds and stores
the input and output data, and can evaluate itself to produce data.
"""

VALID_TYPES = {
    'simulation_model': ['queens.models.simulation_model', 'SimulationModel'],
    'datafit_surrogate_model': ['queens.models.data_fit_surrogate_model', 'DataFitSurrogateModel'],
    'datafit_surrogate_model_mf': [
        'queens.models.data_fit_surrogate_model_mf',
        'MFDataFitSurrogateModel',
    ],
    'multi_fidelity_model': ['queens.models.multifidelity_model', 'MultifidelityModel'],
    'bmfmc_model': ['queens.models.bmfmc_model', 'BMFMCModel'],
    'gaussian': ['queens.models.likelihood_models.gaussian_likelihood', 'GaussianLikelihood'],
    'bmf_gaussian': [
        'queens.models.likelihood_models.bayesian_mf_gaussian_likelihood',
        'BMFGaussianModel',
    ],
    'differentiable_simulation_model_fd': [
        'queens.models.differentiable_simulation_model_fd',
        'DifferentiableSimulationModelFD',
    ],
    'differentiable_simulation_model_adjoint': [
        'queens.models.differentiable_simulation_model_adjoint',
        'DifferentiableSimulationModelAdjoint',
    ],
    'heteroskedastic_gp': [
        'queens.models.surrogate_models.heteroskedastic_GPflow',
        'HeteroskedasticGPModel',
    ],
    'gp_approximation_gpflow': [
        'queens.models.surrogate_models.gp_approximation_gpflow',
        'GPFlowRegressionModel',
    ],
    'gaussian_bayesian_neural_network': [
        'queens.models.surrogate_models.bayesian_neural_network',
        'GaussianBayesianNeuralNetworkModel',
    ],
    'gp_jitted': [
        'queens.models.surrogate_models.gp_approximation_jitted',
        'GPJittedModel',
    ],
    'gp_approximation_gpflow_svgp': [
        'queens.models.surrogate_models.gp_approximation_gpflow_svgp',
        'GPflowSVGPModel',
    ],
    'gaussian_nn': [
        'queens.models.surrogate_models.gaussian_neural_network',
        'GaussianNeuralNetworkModel',
    ],
}
