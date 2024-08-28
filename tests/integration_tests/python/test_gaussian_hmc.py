"""Test HMC Sampler."""

import numpy as np
import pytest
from mock import patch

from queens.distributions.normal import NormalDistribution
from queens.drivers.function_driver import FunctionDriver
from queens.interfaces.job_interface import JobInterface
from queens.iterators.hmc_iterator import HMCIterator
from queens.main import run_iterator
from queens.models.likelihood_models.gaussian_likelihood import GaussianLikelihood
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.schedulers.pool_scheduler import PoolScheduler
from queens.utils.experimental_data_reader import ExperimentalDataReader
from queens.utils.io_utils import load_result


def test_gaussian_hmc(
    tmp_path,
    target_density_gaussian_2d_with_grad,
    _create_experimental_data_zero,
    global_settings,
):
    """Test case for hmc iterator."""
    # Parameters
    x1 = NormalDistribution(mean=[-2.0, 2.0], covariance=[[1.0, 0.0], [0.0, 1.0]])
    parameters = Parameters(x1=x1)

    # Setup iterator
    experimental_data_reader = ExperimentalDataReader(
        file_name_identifier="*.csv",
        csv_data_base_dir=tmp_path,
        output_label="y_obs",
    )
    driver = FunctionDriver(function="patch_for_likelihood")
    scheduler = PoolScheduler(experiment_name=global_settings.experiment_name)
    interface = JobInterface(parameters=parameters, scheduler=scheduler, driver=driver)
    forward_model = SimulationModel(interface=interface)
    model = GaussianLikelihood(
        noise_type="fixed_variance",
        noise_value=1.0,
        experimental_data_reader=experimental_data_reader,
        forward_model=forward_model,
    )
    iterator = HMCIterator(
        seed=42,
        num_samples=10,
        num_burn_in=2,
        num_chains=1,
        use_queens_prior=False,
        progressbar=False,
        result_description={"write_results": True, "plot_results": False, "cov": True},
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    with patch.object(
        GaussianLikelihood, "evaluate_and_gradient", target_density_gaussian_2d_with_grad
    ):
        run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    assert results["mean"].mean(axis=0) == pytest.approx(
        np.array([0.19363280864587615, -1.1303341362165935])
    )
    assert results["var"].mean(axis=0) == pytest.approx([0, 0])
