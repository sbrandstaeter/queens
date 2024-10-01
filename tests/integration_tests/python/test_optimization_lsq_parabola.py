"""Integration test for the Optimization iterator.

This test analyzes the special case of 1 unknown and 1 residual.
"""

import numpy as np

from queens.distributions.free import FreeVariable
from queens.drivers.function_driver import FunctionDriver
from queens.iterators.optimization_iterator import OptimizationIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.schedulers.pool_scheduler import PoolScheduler
from queens.utils.io_utils import load_result


def test_optimization_lsq_parabola(global_settings):
    """Test special case for optimization iterator with the least squares.

    Special case: 1 unknown and 1 residual.
    """
    # Parameters
    x1 = FreeVariable(dimension=1)
    parameters = Parameters(x1=x1)

    # Setup iterator
    driver = FunctionDriver(parameters=parameters, function="parabola_residual")
    scheduler = PoolScheduler(experiment_name=global_settings.experiment_name)
    model = SimulationModel(scheduler=scheduler, driver=driver)
    iterator = OptimizationIterator(
        algorithm="LSQ",
        initial_guess=[0.75],
        result_description={"write_results": True},
        bounds=[float("-inf"), float("inf")],
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    np.testing.assert_allclose(results.x, np.array([+0.3]))
    np.testing.assert_allclose(results.fun, np.array([+0.0]))
