"""Created on Dezember 11th  2017.

@author: jbi
"""
import unittest

import mock
import pytest

from pqueens.database.mongodb import MongoDB
from pqueens.interfaces import from_config_create_interface
from pqueens.interfaces.job_interface import JobInterface


class TestJobInterface(unittest.TestCase):
    def setUp(self):

        uncertain_parameters = {}
        uncertain_parameter = {}
        uncertain_parameter["type"] = "FLOAT"
        uncertain_parameter["size"] = 1
        uncertain_parameter["distribution"] = "uniform"
        uncertain_parameter['lower_bound'] = -3.14159265359
        uncertain_parameter['upper_bound'] = 3.14159265359

        uncertain_parameters['x1'] = uncertain_parameter
        uncertain_parameters['x2'] = uncertain_parameter
        uncertain_parameters['x3'] = uncertain_parameter

        self.config = {}
        self.config['global_settings'] = {}
        self.config['input_file'] = 'test-input-file'
        self.config['global_settings']['experiment_name'] = 'test-experiment'
        self.config['test_interface'] = {'type': 'job_interface', 'resources': 'dummy_resource'}

        self.config['parameters'] = uncertain_parameters

        dummy_resource = {}
        dummy_resource['my_machine'] = {
            'scheduler': 'my_local_scheduler',
            'max-concurrent': 1,
            'max-finished-jobs': 100,
        }
        self.config['restart'] = False

        self.config['database'] = {}
        self.config['database']['address'] = 'localhost:27017'
        self.config['database']['drop_all_existing_dbs'] = True
        self.config['output_dir'] = {}

        self.config['resources'] = {}
        self.config['resources'] = dummy_resource

        self.config['my_local_scheduler'] = {}
        self.config['my_local_scheduler']['experiment_dir'] = 'dummy_dir'
        self.config['my_local_scheduler']['scheduler_type'] = 'standard'
        self.config['my_local_scheduler']['remote'] = False
        self.config['my_local_scheduler']['singularity'] = False

        self.config['driver'] = {}
        self.config['driver']['driver_type'] = 'baci'
        self.config['driver']['driver_params'] = {}
        self.config['driver']['driver_params']['data_processor'] = {}
        self.config['driver']['driver_params']['data_processor']['file_prefix'] = 'test-file-prefix'

    class FakeDB(object):
        database_address = 'localhost:27017'
        database_name = 'test-database'
        database_list = {}
        database_already_existent = False
        drop_all_existing_dbs = True

    db_fake = FakeDB()

    @pytest.mark.unit_tests
    @mock.patch(
        'pqueens.database.mongodb.MongoDB.from_config_create_database', return_value=db_fake
    )
    def test_construction(self, config, **mocks):
        interface = from_config_create_interface('test_interface', self.config)
        # ensure correct type
        self.assertIsInstance(interface, JobInterface)
