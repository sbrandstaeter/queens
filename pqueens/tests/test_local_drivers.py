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

from pqueens.drivers.baci_driver_docker import baci_driver_docker
from pqueens.drivers.baci_driver_docker import run_baci

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

        launch(db_address='mongodb:27017', experiment_name='simple_test',
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

class TestBaciDriverDocker(unittest.TestCase):

    def setUp(self):
        self.mock_job = {}
        self.mock_job['expt_dir'] = 'mock_dir'
        self.mock_job['expt_name'] = 'mock_exp_name'
        self.mock_job['id'] = 1
        self.mock_job['params'] = {'alpha':1, 'beta':2}

        driver_params = {}
        driver_params['input_template'] = 'test_template'
        driver_params['path_to_executable'] = 'mock_path'
        driver_params['path_to_postprocessor'] = 'mock_post_processor'
        driver_params['post_process_options'] = ['mock_post_process_options']
        driver_params['docker_container'] = 'mock_container'
        driver_params['post_post_script'] = 'mock_post_process_script'
        self.mock_job['driver_params'] = driver_params

        self.mock_baci_input_file = self.mock_job['expt_dir'] + '/' + self.mock_job['expt_name'] + '_' + str(self.mock_job['id']) + '.dat'
        self.mock_baci_output_file = self.mock_job['expt_dir'] + '/'+ self.mock_job['expt_name'] + '_' + str(self.mock_job['id'])
        self.mock_baci_cmd = driver_params['path_to_executable'] + ' ' + self.mock_baci_input_file  + ' ' + self.mock_baci_output_file
        self.mock_post_cmd = driver_params['path_to_postprocessor'] + ' ' + 'mock_post_process_options' \
                             + ' --file='+self.mock_baci_output_file + ' --output=mock_dir/mock_exp_name_1_1'
        self.mock_volume_map = {self.mock_job['expt_dir']: {'bind': self.mock_job['expt_dir'], 'mode': 'rw'}}

    # mock os is valid dir
    @mock.patch('os.chdir')
    @mock.patch('pqueens.drivers.baci_driver_docker.inject',return_value='1')
    @mock.patch('pqueens.drivers.baci_driver_docker.run_post_post_processing')
    @mock.patch('pqueens.drivers.baci_driver_docker.run_post_processing')
    @mock.patch('pqueens.drivers.baci_driver_docker.run_baci')
    def test_gen_functionality(self, mock_run_baci, mock_post, mock_post_post,
                               mock_inject, mock_oschdir):

        baci_driver_docker(self.mock_job)
        mock_inject.assert_called_with(self.mock_job['params'],
                                       self.mock_job['driver_params']['input_template'],
                                       self.mock_baci_input_file)
        mock_run_baci.assert_called_with(self.mock_job['driver_params']['docker_container'],
                                         self.mock_baci_cmd, self.mock_volume_map)
        mock_post.assert_called_with(self.mock_job['driver_params']['docker_container'],
                                     self.mock_post_cmd, self.mock_volume_map)

        mock_post_post.assert_called_with(self.mock_job['driver_params']['post_post_script'],
                                          self.mock_baci_output_file)

    # currently it seems impossible to mock the Docker client
    # https://github.com/docker/docker-py/issues/1854
    # so for now we do not test
    # run_baci, run_post_processing, and run_post_post_processing
