"""TODO_doc."""

import numpy as np
from mock import patch


from queens.distributions.normal import NormalDistribution

# fmt: off
from queens.interfaces.direct_python_interface import DirectPythonInterface
from queens.iterators.metropolis_hastings_iterator import MetropolisHastingsIterator
from queens.main import run_iterator
from queens.models.likelihood_models.gaussian_likelihood import GaussianLikelihood

# fmt: on
from queens.iterators.sequential_monte_carlo_iterator import SequentialMonteCarloIterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.utils.experimental_data_reader import ExperimentalDataReader
from queens.utils.io_utils import load_result


def test_metropolis_hastings_multivariate_gaussian(
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
        results['mean'], np.array([[0.7240107551260684, -2.045891088599629]])
    )
    np.testing.assert_allclose(
        results['cov'],
        np.array(
            [
                [
                    [0.30698538649168755, -0.027059991075557278],
                    [-0.027059991075557278, 0.004016365725389411],
                ]
            ]
        ),
    )
