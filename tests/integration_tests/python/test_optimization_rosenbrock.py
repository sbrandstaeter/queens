"""Test suite for integration tests of optimization iterator.

Based on the Rosenbrock test function.
"""

import numpy as np
import pytest

from queens.distributions.free import FreeVariable
from queens.drivers.function_driver import FunctionDriver
from queens.interfaces.job_interface import JobInterface
from queens.iterators.optimization_iterator import OptimizationIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.schedulers.pool_scheduler import PoolScheduler
from queens.utils.io_utils import load_result


@pytest.fixture(
    name="algorithm",
    params=["NELDER-MEAD", "POWELL", "CG", "BFGS", "L-BFGS-B", "TNC", "SLSQP"],
)
def fixture_algorithm(request):
    """TODO_doc."""
    return request.param


def test_optimization_rosenbrock(algorithm, global_settings):
    """Test different solution algorithms in optimization iterator."""
    # Parameters
    x1 = FreeVariable(dimension=1)
    x2 = FreeVariable(dimension=1)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    driver = FunctionDriver(parameters=parameters, function="rosenbrock60")
    scheduler = PoolScheduler(experiment_name=global_settings.experiment_name)
    interface = JobInterface(scheduler=scheduler, driver=driver)
    model = SimulationModel(interface=interface)
    iterator = OptimizationIterator(
        algorithm=algorithm,
        initial_guess=[-3.0, -4.0],
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

    np.testing.assert_allclose(results.x, np.array([+1.0, +1.0]), rtol=1.0e-3)
    np.testing.assert_allclose(results.fun, np.array(+0.0), atol=5.0e-07)
