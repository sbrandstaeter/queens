"""Fixtures needed across all unittests."""
import pytest


@pytest.fixture(scope="session")
def fake_database():
    class FakeDB(object):
        def print_database_information(self, *args, **kwargs):
            print('test')

    # TODO this is super ugly. creation of DB needs te be moved out of
    # driver init to resolve this
    return FakeDB()
