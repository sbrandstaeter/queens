"""Test points iterator."""

import numpy as np
import pytest

from queens.distributions.free import FreeVariable
from queens.example_simulator_functions.rosenbrock60 import rosenbrock60
from queens.interfaces.direct_python_interface import DirectPythonInterface
from queens.iterators.points_iterator import PointsIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.utils.io_utils import load_result


def test_points_iterator(tmp_path, inputs, expected_results, _initialize_global_settings):
    """Integration test for the points iterator."""
    # Parameters
    x1 = FreeVariable(dimension=1)
    x2 = FreeVariable(dimension=1)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup QUEENS stuff
    interface = DirectPythonInterface(function="rosenbrock60", parameters=parameters)
    model = SimulationModel(interface=interface)
    iterator = PointsIterator(
        points=inputs,
        result_description={"write_results": True},
        model=model,
        parameters=parameters,
        global_settings=_initialize_global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=_initialize_global_settings)

    # Load results
    result_file = tmp_path / "dummy_experiment_name.pickle"
    results = load_result(result_file)

    np.testing.assert_array_equal(
        results["output"]["result"],
        expected_results,
    )


def test_points_iterator_failure(_initialize_global_settings):
    """Test failure of the points iterator."""
    inputs = {"x1": [1], "x2": [1, 2]}
    # Parameters
    x1 = FreeVariable(dimension=1)
    x2 = FreeVariable(dimension=1)
    parameters = Parameters(x1=x1, x2=x2)
    # Setup QUEENS stuff
    interface = DirectPythonInterface(function="rosenbrock60", parameters=parameters)
    model = SimulationModel(interface=interface)
    iterator = PointsIterator(
        points=inputs,
        result_description={"write_results": True},
        model=model,
        parameters=parameters,
        global_settings=_initialize_global_settings,
    )

    with pytest.raises(
        ValueError, match="Non-matching number of points for the different parameters: x1: 1, x2: 2"
    ):
        run_iterator(iterator, global_settings=_initialize_global_settings)


@pytest.fixture(name="inputs")
def fixture_inputs():
    """Input fixtures."""
    return {"x1": [1, 2], "x2": [3, 4]}


@pytest.fixture(name="expected_results")
def fixture_expected_results(inputs):
    """Expected results fixture."""
    input_as_array = inputs.copy()
    input_as_array["x1"] = np.array(input_as_array["x1"]).reshape(-1, 1)
    input_as_array["x2"] = np.array(input_as_array["x2"]).reshape(-1, 1)
    return rosenbrock60(**input_as_array)
