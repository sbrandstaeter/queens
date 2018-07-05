'''
Created on July 5th 2018
@author: jbi

'''
import unittest
import mock

from pqueens.drivers.baci_driver_docker import baci_driver_docker

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
