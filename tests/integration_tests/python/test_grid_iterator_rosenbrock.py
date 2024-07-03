"""TODO_doc."""
import numpy as np
import pytest

from queens.distributions.uniform import UniformDistribution
from queens.interfaces.direct_python_interface import DirectPythonInterface
from queens.iterators.grid_iterator import GridIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.utils.io_utils import load_result


def test_grid_iterator(expected_response, expected_grid, global_settings):
    """Integration test for the grid iterator."""
    # Parameters
    x1 = UniformDistribution(lower_bound=-2.0, upper_bound=2.0)
    x2 = UniformDistribution(lower_bound=-2.0, upper_bound=2.0)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup QUEENS stuff
    interface = DirectPythonInterface(function="rosenbrock60", parameters=parameters)
    model = SimulationModel(interface=interface)
    iterator = GridIterator(
        grid_design={
            "x1": {"num_grid_points": 5, "axis_type": "lin", "data_type": "FLOAT"},
            "x2": {"num_grid_points": 5, "axis_type": "lin", "data_type": "FLOAT"},
        },
        result_description={
            "write_results": True,
            "plotting_options": {
                "plot_booleans": [True],
                "plotting_dir": "some/plotting/dir",
                "plot_names": ["grid_plot.eps"],
                "save_bool": [False],
            },
        },
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    np.testing.assert_array_equal(
        results["raw_output_data"]["result"],
        expected_response,
    )

    np.testing.assert_allclose(results["input_data"], expected_grid, rtol=1.0e-3)


@pytest.fixture(name="expected_grid")
def fixture_expected_grid():
    """TODO_doc."""
    input_data = np.array(
        [
            [-2.000, -2.000],
            [-1.000, -2.000],
            [0.000, -2.000],
            [1.000, -2.000],
            [2.000, -2.000],
            [-2.000, -1.000],
            [-1.000, -1.000],
            [0.000, -1.000],
            [1.000, -1.000],
            [2.000, -1.000],
            [-2.000, 0.000],
            [-1.000, 0.000],
            [0.000, 0.000],
            [1.000, 0.000],
            [2.000, 0.000],
            [-2.000, 1.000],
            [-1.000, 1.000],
            [0.000, 1.000],
            [1.000, 1.000],
            [2.000, 1.000],
            [-2.000, 2.000],
            [-1.000, 2.000],
            [0.000, 2.000],
            [1.000, 2.000],
            [2.000, 2.000],
        ]
    )
    return input_data


@pytest.fixture(name="expected_response")
def fixture_expected_response():
    """TODO_doc."""
    expected_response = np.atleast_2d(
        np.array(
            [
                3.609e03,
                9.040e02,
                4.010e02,
                9.000e02,
                3.601e03,
                2.509e03,
                4.040e02,
                1.010e02,
                4.000e02,
                2.501e03,
                1.609e03,
                1.040e02,
                1.000e00,
                1.000e02,
                1.601e03,
                9.090e02,
                4.000e00,
                1.010e02,
                0.000e00,
                9.010e02,
                4.090e02,
                1.040e02,
                4.010e02,
                1.000e02,
                4.010e02,
            ]
        )
    ).T

    return expected_response
