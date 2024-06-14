"""TODO_doc."""

import numpy as np

from queens.distributions.free import FreeVariable
from queens.interfaces.direct_python_interface import DirectPythonInterface
from queens.iterators.optimization_iterator import OptimizationIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.utils.io_utils import load_result


def test_optimization_lsq_rosenbrock(tmp_path, _initialize_global_settings):
    """Test case for optimization iterator with the least squares."""
    # Parameters
    x1 = FreeVariable(dimension=1)
    x2 = FreeVariable(dimension=1)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup QUEENS stuff
    interface = DirectPythonInterface(function="rosenbrock60_residual", parameters=parameters)
    model = SimulationModel(interface=interface)
    iterator = OptimizationIterator(
        algorithm="LSQ",
        initial_guess=[-3.0, -4.0],
        result_description={"write_results": True},
        bounds=[float("-inf"), float("inf")],
        model=model,
        parameters=parameters,
        global_settings=_initialize_global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=_initialize_global_settings)

    # Load results
    results = load_result(tmp_path / f"{_initialize_global_settings.experiment_name}.pickle")

    np.testing.assert_allclose(results.x, np.array([+1.0, +1.0]))
    np.testing.assert_allclose(results.fun, np.array([+0.0, +0.0]))


def test_optimization_lsq_rosenbrock_error(tmp_path, _initialize_global_settings):
    """Test error for optimization iterator with the least squares."""
    # Parameters
    x1 = FreeVariable(dimension=1)
    x2 = FreeVariable(dimension=1)
    x3 = FreeVariable(dimension=1)
    parameters = Parameters(x1=x1, x2=x2, x3=x3)

    # Setup QUEENS stuff
    interface = DirectPythonInterface(function="rosenbrock60_residual_3d", parameters=parameters)
    model = SimulationModel(interface=interface)
    iterator = OptimizationIterator(
        algorithm="LSQ",
        initial_guess=[-3.0, -4.0, -5.0],
        result_description={"write_results": True},
        bounds=[float("-inf"), float("inf")],
        model=model,
        parameters=parameters,
        global_settings=_initialize_global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=_initialize_global_settings)

    # Load results
    results = load_result(tmp_path / f"{_initialize_global_settings.experiment_name}.pickle")

    np.testing.assert_allclose(results.x, np.array([+1.0, +1.0, -0.001039]), rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(results.fun[:2], np.array([+0.0, +0.0]), rtol=1e-07, atol=0)
