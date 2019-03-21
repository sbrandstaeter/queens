import pytest


@pytest.fixture(scope='session')
def job(tmpdir_factory):
    """ Generic job dictionary for testing drivers"""
    job = {}
    job['expt_dir'] = str(tmpdir_factory.mktemp('expt_dir'))
    job['expt_name'] = 'experiment_name'
    job['id'] = 666
    job['params'] = {'alpha': 3.14, 'beta': 2.3}

    return job


@pytest.fixture(scope='session')
def baci_job(job, tmpdir_factory):
    """ Generic job dictionary for testing BACI drivers"""

    baci_dir = tmpdir_factory.mktemp('baci_dir')

    driver_params = {}
    driver_params['input_template'] =  str(tmpdir_factory.mktemp('template_dir').join('template.dat'))
    driver_params['path_to_executable'] = str(baci_dir.join('baci_release'))
    driver_params['path_to_postprocessor'] = str(baci_dir.join('post_processor'))
    driver_params['post_process_options'] = ['post_process_options_1', 'post_process_options_2']
    driver_params['post_post_script'] = str(tmpdir_factory.mktemp('post_post_dir').join('post_post_script.py'))

    job['driver_params'] = driver_params

    return job


@pytest.fixture(scope='session')
def baci_input_file(job):
    """ BACI input file created by inject based on job description"""
    baci_input_file = job['expt_dir'] + '/' + job['expt_name'] + '_' + str(job['id']) + '.dat'
    return baci_input_file


@pytest.fixture(scope='session')
def baci_output_file(job):
    """ BACI output file based on job description"""
    baci_output_file = job['expt_dir'] + '/'+ job['expt_name'] + '_' + str(job['id'])
    return baci_output_file


@pytest.fixture(scope='session')
def baci_cmd(baci_job, baci_input_file, baci_output_file):
        baci_cmd = baci_job['driver_params']['path_to_executable'] + ' ' + baci_input_file + ' ' + baci_output_file
        return baci_cmd


@pytest.fixture(scope='session')
def baci_post_cmds(baci_job, baci_output_file):

    post_cmds = []
    for id, baci_post_process_option in enumerate(baci_job['driver_params']['post_process_options']):
        post_cmd = baci_job['driver_params']['path_to_postprocessor'] +\
                   ' ' + baci_post_process_option +\
                   ' --file=' + baci_output_file +\
                   ' --output=' + baci_job['expt_dir'] + '/' +\
                   baci_job['expt_name'] + '_' + str(baci_job['id']) + '_' + str(id+1)
        post_cmds.append(post_cmd)
    return post_cmds


@pytest.fixture(scope='module')
def baci_docker_job(baci_job, tmpdir_factory):
    """ Job dictionary for testing BACI docker driver"""

    baci_job['driver_params']['docker_container'] =  str(tmpdir_factory.mktemp('docker_container_dir').join('container'))

    return baci_job


@pytest.fixture(scope='module')
def docker_volume_map(job):
        volume_map = {job['expt_dir']: {'bind': job['expt_dir'], 'mode': 'rw'}}
        return volume_map

