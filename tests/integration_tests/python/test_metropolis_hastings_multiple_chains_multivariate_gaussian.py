"""TODO_doc."""

import numpy as np
from mock import patch

from queens.distributions.normal import NormalDistribution
from queens.interfaces.direct_python_interface import DirectPythonInterface
from queens.iterators.metropolis_hastings_iterator import MetropolisHastingsIterator
from queens.iterators.sequential_monte_carlo_iterator import SequentialMonteCarloIterator
from queens.main import run_iterator
from queens.models.likelihood_models.gaussian_likelihood import GaussianLikelihood
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.utils.experimental_data_reader import ExperimentalDataReader
from queens.utils.io_utils import load_result


def test_metropolis_hastings_multiple_chains_multivariate_gaussian(
    tmp_path,
    target_density_gaussian_2d,
    _create_experimental_data_gaussian_2d,
    _initialize_global_settings,
):
    """Test case for Metropolis Hastings iterator."""
    # Parameters
    x1 = NormalDistribution(mean=2.0, covariance=1.0)
    x2 = NormalDistribution(mean=-2.0, covariance=0.01)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup QUEENS stuff
    experimental_data_reader = ExperimentalDataReader(
        file_name_identifier="*.csv",
        csv_data_base_dir=tmp_path,
        output_label="y_obs",
    )
    proposal_distribution = NormalDistribution(mean=[0.0, 0.0], covariance=[[1.0, 0.0], [0.0, 0.1]])
    interface = DirectPythonInterface(function="patch_for_likelihood", parameters=parameters)
    forward_model = SimulationModel(interface=interface)
    model = GaussianLikelihood(
        noise_type="fixed_variance",
        noise_value=1.0,
        nugget_noise_variance=1e-05,
        experimental_data_reader=experimental_data_reader,
        forward_model=forward_model,
    )
    iterator = MetropolisHastingsIterator(
        seed=42,
        num_chains=3,
        num_samples=10,
        num_burn_in=5,
        scale_covariance=1.0,
        result_description={"write_results": True, "plot_results": False, "cov": True},
        proposal_distribution=proposal_distribution,
        model=model,
        parameters=parameters,
        global_settings=_initialize_global_settings,
    )

    # Actual analysis
    # mock methods related to likelihood
    with patch.object(
        SequentialMonteCarloIterator, "eval_log_likelihood", target_density_gaussian_2d
    ):
        with patch.object(
            MetropolisHastingsIterator, "eval_log_likelihood", target_density_gaussian_2d
        ):
            run_iterator(
                iterator,
                global_settings=_initialize_global_settings,
            )

    # Load results
    result_file = tmp_path / "dummy_experiment_name.pickle"
    results = load_result(result_file)

    # note that the analytical solution would be:
    # posterior mean: [0.29378531 -1.97175141]
    # posterior cov: [[0.42937853 0.00282486] [0.00282486 0.00988701]]
    # however, we only have a very inaccurate approximation here:

    np.testing.assert_allclose(
        results['mean'],
        np.array(
            [
                [1.9538477050387937, -1.980155948698723],
                [-0.024456540006756778, -1.9558862932202299],
                [0.8620026644863327, -1.8385635263327393],
            ]
        ),
    )
    np.testing.assert_allclose(
        results['cov'],
        np.array(
            [
                [
                    [0.15127359388133552, 0.07282531084034029],
                    [0.07282531084034029, 0.05171405742642703],
                ],
                [
                    [0.17850797646369507, -0.012342979562824052],
                    [-0.012342979562824052, 0.0023510303586270057],
                ],
                [
                    [0.0019646760257596243, 0.002417903725921208],
                    [0.002417903725921208, 0.002975685737073754],
                ],
            ]
        ),
    )
