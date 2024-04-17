"""TODO_doc."""

import pytest

from queens.distributions.uniform import UniformDistribution
from queens.interfaces.direct_python_interface import DirectPythonInterface
from queens.iterators.lhs_iterator import LHSIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.utils.io_utils import load_result


@pytest.mark.max_time_for_test(20)
def test_branin_latin_hyper_cube(tmp_path, _initialize_global_settings):
    """Test case for latin hyper cube iterator."""
    # Parameters
    x1 = UniformDistribution(lower_bound=-5, upper_bound=10)
    x2 = UniformDistribution(lower_bound=0, upper_bound=15)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup QUEENS stuff
    interface = DirectPythonInterface(function="branin78_hifi", parameters=parameters)
    model = SimulationModel(interface=interface)
    iterator = LHSIterator(
        seed=42,
        num_samples=1000,
        num_iterations=10,
        result_description={"write_results": True, "plot_results": False},
        model=model,
        parameters=parameters,
    )

    # Actual analysis
    run_iterator(iterator)

    # Load results
    result_file = tmp_path / "dummy_experiment_name.pickle"


    results = load_result(result_file)
    assert results["mean"] == pytest.approx(53.17279969296224)
    assert results["var"] == pytest.approx(2581.6502630157715)
