'''
Created on Dezember 11th  2017
@author: jbi

'''
import unittest
import mock
from pqueens.interfaces.job_interface import JobInterface
from pqueens.interfaces.interface import Interface
from pqueens.database.mongodb import MongoDB


class TestJobInterface(unittest.TestCase):
    def setUp(self):

        uncertain_parameters = {}
        uncertain_parameter = {}
        uncertain_parameter["type"] = "FLOAT"
        uncertain_parameter["size"] = 1
        uncertain_parameter["distribution"] = "uniform"
        uncertain_parameter["distribution_parameter"] = [-3.14159265359, 3.14159265359]

        uncertain_parameters['x1'] = uncertain_parameter
        uncertain_parameters['x2'] = uncertain_parameter
        uncertain_parameters['x3'] = uncertain_parameter

        self.config = {}
        self.config['global_settings'] = {}
        self.config['global_settings']['experiment_name'] = 'test-experiment'
        self.config['test_interface'] = {'type': 'job_interface', 'resources': 'dummy_resource'}

        self.config['parameters'] = uncertain_parameters

        dummy_resource = {}
        dummy_resource['my_machine'] = {
            'scheduler': 'my_local_scheduler',
            'max-concurrent': 1,
            'max-finished-jobs': 100,
        }
        self.config['database'] = {}
        self.config['database']['address'] = 'localhost:27017'
        self.config['database']['drop_existing'] = True
        self.config['output_dir'] = {}
        self.config['driver'] = {}
        self.config['driver']['driver_type'] = 'local'
        self.config['driver']['driver_params'] = {}
        self.config['driver']['driver_params']['experiment_dir'] = 'dummy_dir'
        self.config['driver']['driver_params']['restart_from_finished_simulation'] = False

        self.config['resources'] = {}
        self.config['resources'] = dummy_resource

        self.config['my_local_scheduler'] = {}
        self.config['my_local_scheduler']['scheduler_type'] = 'local'

    class FakeDB(object):
        def print_database_information(self, *args, **kwargs):
            print('test')

    db_fake = FakeDB()

    @mock.patch(
        'pqueens.database.mongodb.MongoDB.from_config_create_database', return_value=db_fake
    )
    def test_construction(self, config, **mocks):
        interface = Interface.from_config_create_interface('test_interface', self.config)
        # ensure correct type
        self.assertIsInstance(interface, JobInterface)
