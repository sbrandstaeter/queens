"""Tests for ."""

import numpy as np
import pytest

from queens.distributions.uniform import UniformDistribution
from queens.example_simulator_functions.park91a import park91a_hifi_on_grid
from queens.global_settings import GlobalSettings
from queens.interfaces.direct_python_interface import DirectPythonInterface
from queens.iterators.adaptive_sampling_iterator import AdaptiveSamplingIterator
from queens.iterators.monte_carlo_iterator import MonteCarloIterator
from queens.iterators.sequential_monte_carlo_chopin import SequentialMonteCarloChopinIterator
from queens.main import run_iterator
from queens.models.likelihood_models.gaussian_likelihood import GaussianLikelihood
from queens.models.logpdf_gp_model import LogpdfGPModel
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.utils.io_utils import load_result


@pytest.fixture(
    name="approx_type",
    params=[
        'GPMAP-I',
        'CGPMAP-II',
        'CFBGP',
    ],
)
def fixture_approx_type(request):
    """Approximation type fixture."""
    return request.param


@pytest.fixture(name="parameters")
def fixture_parameters():
    """Parameters fixture."""
    x1 = UniformDistribution(lower_bound=0, upper_bound=1)
    x2 = UniformDistribution(lower_bound=0, upper_bound=1)
    parameters = Parameters(x1=x1, x2=x2)
    return parameters


@pytest.fixture(name="likelihood_model")
def fixture_likelihood_model(parameters):
    """Likelihood model fixture."""
    np.random.seed(42)
    interface = DirectPythonInterface(function=park91a_hifi_on_grid, parameters=parameters)
    forward_model = SimulationModel(interface)

    y_obs = park91a_hifi_on_grid(x1=0.3, x2=0.7)
    noise_var = 1e-4
    y_obs += np.random.randn(y_obs.size) * noise_var ** (1 / 2)

    likelihood_model = GaussianLikelihood(
        forward_model=forward_model,
        noise_type='fixed_variance',
        noise_value=noise_var,
        y_obs=y_obs,
    )
    return likelihood_model


@pytest.fixture(name="expected_mean")
def fixture_expected_mean():
    """Expected mean fixture."""
    expected_mean = {
        'GPMAP-I': [0.30427587, 0.53454306],
        'CGPMAP-II': [0.29874207, 0.74095174],
        'CFBGP': [0.29329803, 0.96108789],
    }
    return expected_mean


@pytest.fixture(name="expected_std")
def fixture_expected_std():
    """Expected standard deviation fixture."""
    expected_std = {
        'GPMAP-I': [0.00101052, 0.03076364],
        'CGPMAP-II': [0.00294576, 0.05681254],
        'CFBGP': [0.00155767, 0.02837052],
    }
    return expected_std


def test_constrained_gp_ip_park(
    tmp_path, approx_type, likelihood_model, parameters, expected_mean, expected_std
):
    """Test."""
    num_steps = 3
    num_new_samples = 4
    num_initial_samples = int(num_new_samples * 2)
    quantile = 0.90
    seed = 41

    if approx_type == "CFBGP":
        num_steps = 2

    experiments_name = approx_type
    global_settings = GlobalSettings(experiment_name=experiments_name, output_dir=tmp_path)

    with global_settings:
        logpdf_gp_model = LogpdfGPModel(
            approx_type=approx_type,
            num_hyper=10,
            num_optimizations=3,
            hmc_burn_in=100,
            hmc_steps=100,
            prior_rate=[1.0e-1, 10.0, 1.0e8],
            prior_gp_mean=-1.0,
            quantile=quantile,
            jitter=1.0e-16,
        )

        initial_train_iterator = MonteCarloIterator(
            model=None,
            parameters=parameters,
            global_settings=global_settings,
            seed=seed,
            num_samples=num_initial_samples,
        )

        solving_iterator = SequentialMonteCarloChopinIterator(
            model=logpdf_gp_model,
            parameters=parameters,
            global_settings=global_settings,
            seed=42,
            waste_free=True,
            feynman_kac_model="adaptive_tempering",
            max_feval=1_000_000_000,
            num_particles=3000,
            num_rejuvenation_steps=30,
            resampling_method='residual',
            resampling_threshold=0.5,
            result_description={},
        )

        adaptive_sampling_iterator = AdaptiveSamplingIterator(
            model=logpdf_gp_model,
            parameters=parameters,
            global_settings=global_settings,
            likelihood_model=likelihood_model,
            initial_train_iterator=initial_train_iterator,
            solving_iterator=solving_iterator,
            num_new_samples=num_new_samples,
            num_steps=num_steps,
        )

        run_iterator(adaptive_sampling_iterator, global_settings)

        results = load_result(tmp_path / (experiments_name + '.pickle'))

        particles = results['particles'][-1]
        weights = results['weights'][-1]

        mean = np.average(particles, weights=weights, axis=0)
        std = np.average((particles - mean) ** 2, weights=weights, axis=0) ** (1 / 2)

        np.testing.assert_almost_equal(mean, expected_mean[approx_type])
        np.testing.assert_almost_equal(std, expected_std[approx_type])
