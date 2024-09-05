"""TODO_doc."""

import pytest
from mock import patch

from queens.distributions.normal import NormalDistribution
from queens.drivers.function_driver import FunctionDriver
from queens.iterators.metropolis_hastings_iterator import MetropolisHastingsIterator
from queens.main import run_iterator
from queens.models.likelihood_models.gaussian_likelihood import GaussianLikelihood
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.schedulers.pool_scheduler import PoolScheduler
from queens.utils.experimental_data_reader import ExperimentalDataReader
from queens.utils.io_utils import load_result


def test_gaussian_metropolis_hastings(
    tmp_path,
    target_density_gaussian_1d,
    _create_experimental_data_gaussian_1d,
    global_settings,
):
    """Test case for Metropolis Hastings iterator."""
    # Parameters
    x = NormalDistribution(mean=2, covariance=1)
    parameters = Parameters(x=x)

    # Setup iterator
    experimental_data_reader = ExperimentalDataReader(
        file_name_identifier="*.csv",
        csv_data_base_dir=tmp_path,
        output_label="y_obs",
    )
    proposal_distribution = NormalDistribution(mean=0.0, covariance=1.0)
    driver = FunctionDriver(parameters=parameters, function="patch_for_likelihood")
    scheduler = PoolScheduler(experiment_name=global_settings.experiment_name)
    forward_model = SimulationModel(scheduler=scheduler, driver=driver)
    model = GaussianLikelihood(
        noise_type="fixed_variance",
        noise_value=1.0,
        nugget_noise_variance=1e-05,
        experimental_data_reader=experimental_data_reader,
        forward_model=forward_model,
    )
    iterator = MetropolisHastingsIterator(
        seed=42,
        num_samples=10,
        num_burn_in=5,
        scale_covariance=1.0,
        result_description={"write_results": True, "plot_results": False, "cov": True},
        proposal_distribution=proposal_distribution,
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    with patch.object(
        MetropolisHastingsIterator, "eval_log_likelihood", target_density_gaussian_1d
    ):
        run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    # note that the analytical solution would be:
    # posterior mean: [1.]
    # posterior var: [0.5]
    # posterior std: [0.70710678]
    # however, we only have a very inaccurate approximation here:
    assert results["mean"] == pytest.approx(1.046641592648936)
    assert results["var"] == pytest.approx(0.3190199514534667)
