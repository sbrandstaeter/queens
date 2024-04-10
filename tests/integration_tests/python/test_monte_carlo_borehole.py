"""TODO_doc."""
import pytest

from queens.distributions.uniform import UniformDistribution
from queens.global_settings import GlobalSettings
from queens.interfaces.direct_python_interface import DirectPythonInterface
from queens.iterators.monte_carlo_iterator import MonteCarloIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.utils.io_utils import load_result


def test_monte_carlo_borehole(tmp_path):
    """Test case for Monte Carlo iterator."""
    # Global settings
    experiment_name = "monte_carlo_borehole"
    output_dir = tmp_path

    with GlobalSettings(experiment_name=experiment_name, output_dir=output_dir, debug=False) as gs:
        # Parameters
        rw = UniformDistribution(lower_bound=0.05, upper_bound=0.15)
        r = UniformDistribution(lower_bound=100, upper_bound=50000)
        tu = UniformDistribution(lower_bound=63070, upper_bound=115600)
        hu = UniformDistribution(lower_bound=990, upper_bound=1110)
        tl = UniformDistribution(lower_bound=63.1, upper_bound=116)
        hl = UniformDistribution(lower_bound=700, upper_bound=820)
        l = UniformDistribution(lower_bound=1120, upper_bound=1680)
        kw = UniformDistribution(lower_bound=9855, upper_bound=12045)
        parameters = Parameters(rw=rw, r=r, tu=tu, hu=hu, tl=tl, hl=hl, l=l, kw=kw)

        # Setup QUEENS stuff
        interface = DirectPythonInterface(function="borehole83_lofi", parameters=parameters)
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
    assert results["mean"] == pytest.approx(60.4546131041304)
    assert results["var"] == pytest.approx(1268.1681250046817)
