"""Fixtures needed across unit_tests."""

import pytest

from queens.global_settings import GlobalSettings


@pytest.fixture(name="_initialize_global_settings")
def fixture_initialize_global_settings(tmp_path):
    """Initialize GlobalSettings object."""

    # Setup and initialize global settings
    global_settings = GlobalSettings(experiment_name="dummy_experiment_name", output_dir=tmp_path)

    # wrap the tests in a global settings context
    with global_settings:
        yield global_settings
