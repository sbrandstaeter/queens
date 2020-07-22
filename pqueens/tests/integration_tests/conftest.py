''' Collect fixtures used by the function tests. '''
import os

import pytest


@pytest.fixture(scope='session')
def inputdir():
    ''' Return the path to the json input-files of the function test. '''
    dirpath = os.path.dirname(__file__)
    input_files_path = os.path.join(dirpath, 'queens_input_files')
    return input_files_path
