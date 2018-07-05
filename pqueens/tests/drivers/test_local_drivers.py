'''
Created on December 26th 2017
@author: jbi

'''
import argparse
import unittest
import mock

from pqueens.drivers.gen_driver_local import launch
from pqueens.drivers.gen_driver_local import parse_args
from pqueens.drivers.gen_driver_local import main as driver_main


class TestGenDriverLocal(unittest.TestCase):
    # create mock for Database, specifically for load and save functions
    @mock.patch.multiple('pqueens.database.mongodb.MongoDB', __init__=mock.Mock(return_value=None), load=mock.DEFAULT, save=mock.DEFAULT)
    # create mock for respective local driver
    @mock.patch('pqueens.drivers.gen_driver_local.baci_driver_native', return_value='1.5')
    def test_baci_native(self, mock_driver, **mocks):
        mock_load = mocks['load']
        mock_save = mocks['save']
        mock_load.return_value = {'id': 1,
                                  'driver_type'  : 'baci_native',
                                  'status'       : 'new',
                                  'submit time'  : 15,
                                  'start time'   : None,
                                  'end time'     : None}

        mock_save.return_value = 1

        launch(db_address='localhost:27017', experiment_name='simple_test',
               batch=1, job_id=1)

        mock_driver.assert_called_with({'id': 1,
                                        'driver_type'  : 'baci_native',
                                        'status'       : 'complete',
                                        'submit time'  : 15,
                                        'start time'   : mock.ANY,
                                        'end time'     : mock.ANY,
                                        'result'       : '1.5'})

    @mock.patch.multiple('pqueens.database.mongodb.MongoDB', __init__=mock.Mock(return_value=None),load=mock.DEFAULT, save=mock.DEFAULT)
    @mock.patch('pqueens.drivers.gen_driver_local.baci_driver_docker', return_value='1.5')
    def test_baci_docker(self, mock_driver, **mocks):
        mock_load = mocks['load']
        mock_save = mocks['save']
        mock_load.return_value = {'id': 1,
                                  'driver_type'  : 'baci_docker',
                                  'status'       : 'new',
                                  'submit time'  : 15,
                                  'start time'   : None,
                                  'end time'     : None}

        mock_save.return_value = 1

        launch(db_address='localhost:27017', experiment_name='simple_test',
               batch=1, job_id=1)

        mock_driver.assert_called_with({'id': 1,
                                        'driver_type'  : 'baci_docker',
                                        'status'       : 'complete',
                                        'submit time'  : 15,
                                        'start time'   : mock.ANY,
                                        'end time'     : mock.ANY,
                                        'result'       : '1.5'})

    @mock.patch.multiple('pqueens.database.mongodb.MongoDB',__init__=mock.Mock(return_value=None), load=mock.DEFAULT, save=mock.DEFAULT)
    @mock.patch('pqueens.drivers.gen_driver_local.python_driver_vector_interface', return_value='1.5')
    def test_python_vector(self, mock_driver, **mocks):
        mock_load = mocks['load']
        mock_save = mocks['save']
        mock_load.return_value = {'id': 1,
                                  'driver_type'  : 'python_vector_interface',
                                  'status'       : 'new',
                                  'submit time'  : 15,
                                  'start time'   : None,
                                  'end time'     : None}

        mock_save.return_value = 1

        launch(db_address='localhost:27017', experiment_name='simple_test',
               batch=1, job_id=1)

        mock_driver.assert_called_with({'id': 1,
                                        'driver_type'  : 'python_vector_interface',
                                        'status'       : 'complete',
                                        'submit time'  : 15,
                                        'start time'   : mock.ANY,
                                        'end time'     : mock.ANY,
                                        'result'       : '1.5'})

    @mock.patch.multiple('pqueens.database.mongodb.MongoDB', __init__=mock.Mock(return_value=None), load=mock.DEFAULT, save=mock.DEFAULT)
    @mock.patch('pqueens.drivers.gen_driver_local.python_driver_vector_interface', return_value='1.5')
    def test_unknown_driver(self, mock_driver, **mocks):
        mock_load = mocks['load']
        mock_save = mocks['save']
        mock_load.return_value = {'id': 1,
                                  'driver_type'  : 'wrong_driver',
                                  'status'       : 'new',
                                  'submit time'  : 15,
                                  'start time'   : None,
                                  'end time'     : None}

        mock_save.return_value = 1
        launch(db_address='localhost:27017', experiment_name='simple_test',
               batch=1, job_id=1)

        mock_save.assert_called_with({'id': 1,
                                      'driver_type'  : 'wrong_driver',
                                      'status'       : 'broken',
                                      'submit time'  : 15,
                                      'start time'   : mock.ANY,
                                      'end time'     : mock.ANY},
                                     'simple_test',
                                     'jobs',
                                     1,
                                     {'id': 1})

    def test_option_parser(self):
        args = ['--experiment_name', 'simple_test',
                '--db_address', 'localhost:27017',
                '--job_id', '1',
                '--batch', '1']
        parsed_args = parse_args(args)
        self.assertEqual(parsed_args.db_address, 'localhost:27017')
        self.assertEqual(parsed_args.experiment_name, 'simple_test')
        self.assertEqual(parsed_args.batch, '1')
        self.assertEqual(parsed_args.job_id, 1)
        # test if we get the correct error messages when passing the wrong options
        with self.assertRaises(RuntimeError):
            parsed_args = parse_args(['--db_address', 'localhost:27017',
                                      '--job_id', '1', '--batch', '1'])
        with self.assertRaises(RuntimeError):
            parsed_args = parse_args(['--experiment_name', 'simple_test',
                                      '--job_id', '1', '--batch', '1'])

        with self.assertRaises(RuntimeError):
            parsed_args = parse_args(['--db_address', 'localhost:27017',
                                      '--experiment_name', 'simple_test',
                                      '--job_id', '1'])

        with self.assertRaises(RuntimeError):
            parsed_args = parse_args(['--db_address', 'localhost:27017',
                                      '--experiment_name', 'simple_test',
                                      '--batch', '1'])

    @mock.patch('pqueens.drivers.gen_driver_local.parse_args')
    @mock.patch('pqueens.drivers.gen_driver_local.launch')
    def test_main(self,mock_launcher, mock_parser):
        args = ['--experiment_name', 'simple_test',
                '--db_address', 'localhost:27017',
                '--job_id', '1',
                '--batch', '1']
        parsed_args = argparse.Namespace()
        parsed_args.experiment_name = 'simple_test'
        parsed_args.db_address = 'localhost:27017'
        parsed_args.job_id = 1
        parsed_args.batch = '1'
        mock_parser.return_value = parsed_args

        driver_main(args)
        mock_parser.assert_called_with(args)
        mock_launcher.assert_called_with('localhost:27017', 'simple_test', '1', 1)
