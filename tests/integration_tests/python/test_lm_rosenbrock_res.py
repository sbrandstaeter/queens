#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Integration test for the Levenberg Marquardt iterator."""

import numpy as np

from queens.distributions.free_variable import FreeVariable
from queens.drivers.function import Function
from queens.iterators.least_squares import LeastSquares
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.schedulers.pool import Pool
from queens.utils.io import load_result


def test_lm_rosenbrock_res(global_settings):
    """Test case for Levenberg Marquardt iterator."""
    # Parameters
    x1 = FreeVariable(dimension=1)
    x2 = FreeVariable(dimension=1)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    driver = Function(parameters=parameters, function="rosenbrock60_residual")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    model = Simulation(scheduler=scheduler, driver=driver)
    iterator = LeastSquares(
        model=model,
        parameters=parameters,
        global_settings=global_settings,
        initial_guess=[0.9, 0.9],
        result_description={"write_results": True, "plot_results": True},
        verbose_output=False,
        max_feval=99,
        algorithm="lm",
        jac_method="2-point",
        jac_rel_step=1e-05,
        objective_and_jacobian=True,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    np.testing.assert_allclose(results.x, np.array([+1.0, +1.0]), rtol=1.0e-5)
    np.testing.assert_allclose(results.fun, np.array([+0.0, +0.0]))
