"""Fixtures needed across unit_tests."""

import pytest

from queens.global_settings import GlobalSettings


@pytest.fixture(name="dummy_global_settings")
def fixture_dummy_global_settings(tmp_path):
    """Dummy GlobalSettings object."""
    # Setup global settings
    global_settings = GlobalSettings(experiment_name="dummy_experiment_name", output_dir=tmp_path)
    with global_settings:
        yield global_settings