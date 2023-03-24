"""Fixtures needed across all unit_tests."""
import logging

import pytest

_logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def fake_database():
    """TODO_doc."""

    class FakeDB(object):
        def print_database_information(self, *args, **kwargs):
            _logger.info('test')

    # TODO this is super ugly. creation of DB needs te be moved out of
    # driver init to resolve this
    return FakeDB()


@pytest.fixture(name="test_path")
def fixture_test_path(tmp_path):
    """Convert *tmp_path* to *pathlib* object."""
    return tmp_path
