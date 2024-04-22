"""Fixtures needed across unit_tests."""

import pytest

from queens.global_settings import GlobalSettings
from queens.models.simulation_model import SimulationModel


@pytest.fixture(name="_initialize_global_settings")
def fixture_initialize_global_settings(tmp_path):
    """Initialize GlobalSettings object."""
    # Setup and initialize global settings
    global_settings = GlobalSettings(experiment_name="dummy_experiment_name", output_dir=tmp_path)

    # wrap the tests in a global settings context
    with global_settings:
        yield global_settings


@pytest.fixture(name="result_description")
def fixture_result_description():
    """Fixture for a dummy result description."""
    description = {"write_results": True}
    return description


@pytest.fixture(name="dummy_simulation_model")
def fixture_dummy_simulation_model():
    """Fixture for dummy model."""
    interface = 'my_dummy_interface'
    model = SimulationModel(interface)
    return model
