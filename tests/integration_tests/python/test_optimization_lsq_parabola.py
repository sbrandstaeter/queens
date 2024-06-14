"""TODO_doc."""
import numpy as np

from queens.distributions.free import FreeVariable
from queens.interfaces.direct_python_interface import DirectPythonInterface
from queens.iterators.optimization_iterator import OptimizationIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.utils.io_utils import load_result


def test_optimization_lsq_parabola(tmp_path, _initialize_global_settings):
    """Test special case for optimization iterator with the least squares.

    Special case: 1 unknown and 1 residual.
    """
    # Parameters
    x1 = FreeVariable(dimension=1)
    parameters = Parameters(x1=x1)

    # Setup QUEENS stuff
    interface = DirectPythonInterface(function="parabola_residual", parameters=parameters)
    model = SimulationModel(interface=interface)
    iterator = OptimizationIterator(
        algorithm="LSQ",
        initial_guess=[0.75],
        result_description={"write_results": True},
        bounds=[float("-inf"), float("inf")],
        model=model,
        parameters=parameters,
        global_settings=_initialize_global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=_initialize_global_settings)

    # Load results
    result_file = tmp_path / "dummy_experiment_name.pickle"
    results = load_result(result_file)

    np.testing.assert_allclose(results.x, np.array([+0.3]))
    np.testing.assert_allclose(results.fun, np.array([+0.0]))
