"""TODO_doc."""
import numpy as np
import pytest

from queens.distributions.uniform import UniformDistribution
from queens.interfaces.direct_python_interface import DirectPythonInterface
from queens.iterators.monte_carlo_iterator import MonteCarloIterator
from queens.main import run_iterator
from queens.models import HeteroskedasticGPModel
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.utils.io_utils import load_result


@pytest.mark.max_time_for_test(30)
def test_branin_gpflow_heteroskedastic(
    tmp_path, expected_mean, expected_var, _initialize_global_settings
):
    """Test case for GPflow based heteroskedastic model."""
    # Parameters
    x1 = UniformDistribution(lower_bound=-5, upper_bound=10)
    x2 = UniformDistribution(lower_bound=0, upper_bound=15)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup QUEENS stuff
    interface = DirectPythonInterface(function="branin78_hifi", parameters=parameters)
    model = SimulationModel(interface=interface)
    training_iterator = MonteCarloIterator(
        seed=42,
        num_samples=100,
        model=model,
        parameters=parameters,
        global_settings=_initialize_global_settings,
    )
    model = HeteroskedasticGPModel(
        eval_fit=None,
        error_measures=[
            "sum_squared",
            "mean_squared",
            "root_mean_squared",
            "sum_abs",
            "mean_abs",
            "abs_max",
        ],
        num_posterior_samples=None,
        num_inducing_points=30,
        num_epochs=100,
        adams_training_rate=0.1,
        random_seed=1,
        num_samples_stats=1000,
        training_iterator=training_iterator,
    )

    iterator = MonteCarloIterator(
        seed=44,
        num_samples=10,
        result_description={
            "write_results": True,
            "plot_results": False,
            "bayesian": False,
            "num_support_points": 10,
            "estimate_all": False,
        },
        model=model,
        parameters=parameters,
        global_settings=_initialize_global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=_initialize_global_settings)

    # Load results
    result_file = tmp_path / "dummy_experiment_name.pickle"
    results = load_result(result_file)

    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["result"], expected_mean, decimal=2
    )
    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["variance"], expected_var, decimal=2
    )


@pytest.fixture(name="expected_mean")
def fixture_expected_mean():
    """TODO_doc."""
    mean = np.array(
        [
            [
                5.12898,
                4.07712,
                10.22693,
                2.55123,
                4.56184,
                2.45215,
                2.56100,
                3.32164,
                7.84209,
                6.96919,
            ]
        ]
    ).T
    return mean


@pytest.fixture(name="expected_var")
def fixture_expected_var():
    """TODO_doc."""
    var = np.array(
        [
            [
                1057.66078,
                4802.57196,
                1298.08163,
                1217.39827,
                456.70756,
                13143.74176,
                8244.52203,
                21364.59699,
                877.14343,
                207.58535,
            ]
        ]
    ).T
    return var
