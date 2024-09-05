"""TODO_doc."""

import numpy as np
import pandas as pd

from queens.distributions.free import FreeVariable
from queens.drivers.function_driver import FunctionDriver
from queens.iterators.lm_iterator import LMIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.schedulers.pool_scheduler import PoolScheduler


def test_lm_rosenbrock_res(global_settings):
    """Test case for Levenberg Marquardt iterator."""
    # Parameters
    x1 = FreeVariable(dimension=1)
    x2 = FreeVariable(dimension=1)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    driver = FunctionDriver(parameters=parameters, function="rosenbrock60_residual")
    scheduler = PoolScheduler(experiment_name=global_settings.experiment_name)
    model = SimulationModel(scheduler=scheduler, driver=driver)
    iterator = LMIterator(
        jac_rel_step=1e-05,
        jac_abs_step=0.001,
        max_feval=99,
        init_reg=0.01,
        update_reg="grad",
        convergence_tolerance=1e-06,
        initial_guess=[0.9, 0.9],
        result_description={"write_results": True, "plot_results": True},
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    result_file = global_settings.result_file(".csv")
    data = pd.read_csv(
        result_file,
        sep="\t",
    )

    params = data.get("params").tail(1)
    dfparams = params.str.extractall(r"([+-]?\d+\.\d*e?[+-]?\d*)")
    dfparams = dfparams.astype(float)
    numpyparams = dfparams.to_numpy()

    np.testing.assert_allclose(numpyparams, np.array([[+1.0], [+1.0]]), rtol=1.0e-5)

    assert global_settings.result_file(".html").is_file()
