"""Test job interface."""
import pytest

import pqueens.database.mongodb
from pqueens.interfaces import from_config_create_interface
from pqueens.interfaces.job_interface import JobInterface


@pytest.fixture()
def jobinterface_config():
    """Test configuration for testing the JobInterface class."""
    uncertain_parameters = {}
    uncertain_parameter = {}
    uncertain_parameter["type"] = "FLOAT"
    uncertain_parameter["dimension"] = 1
    uncertain_parameter["type"] = "uniform"
    uncertain_parameter['lower_bound'] = -3.14159265359
    uncertain_parameter['upper_bound'] = 3.14159265359

    uncertain_parameters['x1'] = uncertain_parameter
    uncertain_parameters['x2'] = uncertain_parameter
    uncertain_parameters['x3'] = uncertain_parameter

    config = {}
    config['global_settings'] = {}
    config['input_file'] = 'test-input-file'
    config['global_settings']['experiment_name'] = 'test-experiment'
    config['test_interface'] = {
        'type': 'job_interface',
        'resources': 'dummy_resource',
        'driver_name': 'dummy_driver_name',
    }

    config['parameters'] = uncertain_parameters

    dummy_resource = {}
    dummy_resource['my_machine'] = {
        'scheduler_name': 'my_local_scheduler',
        'max-concurrent': 1,
        'max-finished-jobs': 100,
    }
    config['restart'] = False

    config['database'] = {}
    config['database']['address'] = 'localhost:27017'
    config['database']['drop_all_existing_dbs'] = True
    config['output_dir'] = {}

    config['resources'] = {}
    config['resources'] = dummy_resource

    config['my_local_scheduler'] = {}
    config['my_local_scheduler']['type'] = 'standard'
    config['my_local_scheduler']['remote'] = False
    config['my_local_scheduler']['singularity'] = False

    config['driver'] = {}
    config['driver']['driver_type'] = 'mpi'
    config['driver']['data_processor'] = {}
    config['driver']['data_processor']['file_prefix'] = 'test-file-prefix'

    return config


@pytest.fixture()
def mock_db():
    """Mock the database class."""

    class MockDB:
        """Mock database for testing."""

        database_address = 'localhost:27017'
        database_name = 'test-database'
        database_list = {}
        database_already_existent = False
        drop_all_existing_dbs = True

    db_mock = MockDB()
    return db_mock


def test_construction(jobinterface_config, mock_db, monkeypatch):
    """Test construction of job interface."""

    def mock_from_config_create_database():
        """Mock from config_create_database to return an instance of MockDB."""
        return mock_db

    monkeypatch.setattr(
        pqueens.database.mongodb.MongoDB,
        "from_config_create_database",
        mock_from_config_create_database,
    )
    interface = from_config_create_interface('test_interface', jobinterface_config)
    # ensure correct type
    assert isinstance(interface, JobInterface)
