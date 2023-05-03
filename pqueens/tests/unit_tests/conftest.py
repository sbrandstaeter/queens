"""Fixtures needed across all unit_tests."""
import logging

import pytest

_logger = logging.getLogger(__name__)


@pytest.fixture(name="test_path")
def fixture_test_path(tmp_path):
    """Convert *tmp_path* to *pathlib* object."""
    return tmp_path
