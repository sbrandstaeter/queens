'''
Created on Dezember 11th  2017
@author: jbi

'''
import unittest
import mock
from pqueens.interfaces.job_interface import JobInterface
from pqueens.interfaces.interface import Interface

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
        self.config['experiment-name'] = 'test-experiment'
        self.config['test_interface'] = {'type':'job_interface',
                                         'resources': 'dummy_resource'}

        self.config['parameters'] = uncertain_parameters

        dummy_resource = {}
        dummy_resource['my_machine'] = {'scheduler': 'my_local_scheduler', 'max-concurrent':5,
                                        'max-finished-jobs' : 100}
        self.config['database'] = {}
        self.config['database']['address'] = 'localhost:27017'
        self.config['output_dir'] = {}
        self.config['driver'] = {}
        self.config['driver']['driver_type'] = 'local'
        self.config['driver']['driver_params'] = {}
        self.config['driver']['driver_params']['experiment_dir'] = 'dummy_dir'

        self.config['resources'] = {}
        self.config['resources'] = dummy_resource

        self.config['my_local_scheduler'] = {}
        self.config['my_local_scheduler']['scheduler_type'] = 'local'

    @mock.patch.multiple('pqueens.database.mongodb.MongoDB', __init__=mock.Mock(return_value=None), load=mock.DEFAULT, save=mock.DEFAULT)
    def test_construction(self, **mocks):
        interface = Interface.from_config_create_interface('test_interface', self.config)
        # ensure correct type
        self.assertIsInstance(interface, JobInterface)