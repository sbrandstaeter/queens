"""Fixtures needed across unit_tests."""

import pytest

from pqueens.global_settings import GlobalSettings


@pytest.fixture(name="dummy_global_settings")
def dummy_global_settings(tmp_path):
    """Dummy GlobalSettings object."""
    # Setup global settings
    global_settings = GlobalSettings(experiment_name="dummy_experiment_name", output_dir=tmp_path)
    global_settings.__enter__()  # pylint: disable=unnecessary-dunder-call
    yield global_settings

    # Tear down global settings
    print("\ntearing down")
    global_settings.__exit__(None, None, None)
