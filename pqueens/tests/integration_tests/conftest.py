''' Collect fixtures used by the function tests. '''
import os
import pytest
from pathlib import Path


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


@pytest.fixture()
def set_baci_links_for_gitlab_runner(config_dir):
    """ Set symbolic links for baci on testing machine"""
    dst_baci = os.path.join(config_dir, 'baci-release')
    dst_drt_monitor = os.path.join(config_dir, 'post_drt_monitor')
    home = Path.home()
    src_baci = Path.joinpath(home, 'workspace/build/baci-release')
    src_drt_monitor = Path.joinpath(home, 'workspace/build/post_drt_monitor')
    return dst_baci, dst_drt_monitor, src_baci, src_drt_monitor
