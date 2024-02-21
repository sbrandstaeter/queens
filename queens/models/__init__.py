"""Models.

The model package contains several types of models to be used in the
context of UQ. Within QUEENS, the model class of object holds and stores
the input and output data, and can evaluate itself to produce data.
"""

from queens.models.bmfmc_model import BMFMCModel
from queens.models.differentiable_simulation_model_adjoint import (
    DifferentiableSimulationModelAdjoint,
)
from queens.models.differentiable_simulation_model_fd import DifferentiableSimulationModelFD
from queens.models.likelihood_models.bayesian_mf_gaussian_likelihood import BMFGaussianModel
from queens.models.likelihood_models.gaussian_likelihood import GaussianLikelihood
from queens.models.simulation_model import SimulationModel
from queens.models.surrogate_models.bayesian_neural_network import (
    GaussianBayesianNeuralNetworkModel,
)
from queens.models.surrogate_models.gaussian_neural_network import GaussianNeuralNetworkModel
from queens.models.surrogate_models.gp_approximation_gpflow import GPFlowRegressionModel
from queens.models.surrogate_models.gp_approximation_gpflow_svgp import GPflowSVGPModel
from queens.models.surrogate_models.gp_approximation_jitted import GPJittedModel
from queens.models.surrogate_models.gp_heteroskedastic_gpflow import HeteroskedasticGPModel

VALID_TYPES = {
    'simulation_model': SimulationModel,
    'bmfmc_model': BMFMCModel,
    'gaussian': GaussianLikelihood,
    'bmf_gaussian': BMFGaussianModel,
    'differentiable_simulation_model_fd': DifferentiableSimulationModelFD,
    'differentiable_simulation_model_adjoint': DifferentiableSimulationModelAdjoint,
    'heteroskedastic_gp': HeteroskedasticGPModel,
    'gp_approximation_gpflow': GPFlowRegressionModel,
    'gaussian_bayesian_neural_network': GaussianBayesianNeuralNetworkModel,
    'gp_jitted': GPJittedModel,
    'gp_approximation_gpflow_svgp': GPflowSVGPModel,
    'gaussian_nn': GaussianNeuralNetworkModel,
}
