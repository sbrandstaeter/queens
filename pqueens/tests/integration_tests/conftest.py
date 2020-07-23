''' Collect fixtures used by the function tests. '''
import os

import pytest


@pytest.fixture(scope='session')
def inputdir():
    ''' Return the path to the json input-files of the function test. '''
    dirpath = os.path.dirname(__file__)
    input_files_path = os.path.join(dirpath, 'queens_input_files')
    return input_files_path


@pytest.fixture(scope='session')
def third_party_inputs():
    ''' Return the path to the json input-files of the function test. '''
    dirpath = os.path.dirname(__file__)
    input_files_path = os.path.join(dirpath, 'third_party_input_files')
    return input_files_path


@pytest.fixture(scope='session')
def config_dir():
    ''' Return the path to the json input-files of the function test. '''
    dirpath = os.path.dirname(__file__)
    config_dir_path = os.path.join(dirpath, '../../../config')
    return config_dir_path
