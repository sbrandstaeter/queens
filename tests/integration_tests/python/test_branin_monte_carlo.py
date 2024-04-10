"""TODO_doc."""
import pickle


import pytest

from queens.distributions.uniform import UniformDistribution
from queens.global_settings import GlobalSettings
from queens.interfaces.direct_python_interface import DirectPythonInterface
from queens.iterators.monte_carlo_iterator import MonteCarloIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.utils.io_utils import load_result


def test_branin_monte_carlo(tmp_path):
    """Test case for Monte Carlo iterator."""
    # Global settings
    experiment_name = "monte_carlo_branin"
    output_dir = tmp_path

    with GlobalSettings(experiment_name=experiment_name, output_dir=output_dir, debug=False) as gs:
        # Parameters
        x1 = UniformDistribution(lower_bound=-5, upper_bound=10)
        x2 = UniformDistribution(lower_bound=0, upper_bound=15)
        parameters = Parameters(x1=x1, x2=x2)

        # Setup QUEENS stuff
        interface = DirectPythonInterface(function="branin78_hifi", parameters=parameters)
        model = SimulationModel(interface=interface)
        iterator = MonteCarloIterator(
            seed=42,
            num_samples=1000,
            result_description={"write_results": True, "plot_results": False},
            model=model,
            parameters=parameters,
        )

        # Actual analysis
        run_iterator(iterator)

        # Load results
        result_file = gs.output_dir / f"{gs.experiment_name}.pickle"

    results = load_result(result_file)
    assert results["mean"] == pytest.approx(55.81419875080866)
    assert results["var"] == pytest.approx(2754.1188056842070)
