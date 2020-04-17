""" Fixtures needed for testing the Driver classes. """
import pytest
from pqueens.drivers.ansys_driver_native import AnsysDriverNative
from pqueens.database.mongodb import MongoDB


@pytest.fixture(scope='session')
def job(tmpdir_factory):
    """ Generic job dictionary for testing drivers"""
    job_dict = dict()
    job_dict['expt_dir'] = str(tmpdir_factory.mktemp('expt_dir'))
    job_dict['expt_name'] = 'experiment_name'
    job_dict['id'] = 666
    job_dict['params'] = {'alpha': 3.14, 'beta': 2.3}

    return job_dict


@pytest.fixture(scope='session')
def baci_job(job, tmpdir_factory):
    """ Generic job dictionary for testing BACI drivers"""

    baci_dir = tmpdir_factory.mktemp('baci_dir')

    driver_params = {}
    driver_params['input_template'] = str(
        tmpdir_factory.mktemp('template_dir').join('template.dat')
    )
    driver_params['path_to_executable'] = str(baci_dir.join('baci_release'))
    driver_params['path_to_postprocessor'] = str(baci_dir.join('post_processor'))
    driver_params['post_process_options'] = ['post_process_options_1', 'post_process_options_2']
    driver_params['post_post_script'] = str(
        tmpdir_factory.mktemp('post_post_dir').join('post_post_script.py')
    )

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
    baci_output_file = job['expt_dir'] + '/' + job['expt_name'] + '_' + str(job['id'])
    return baci_output_file


@pytest.fixture(scope='session')
def baci_cmd(baci_job, baci_input_file, baci_output_file):
    baci_cmd = (
        baci_job['driver_params']['path_to_executable']
        + ' '
        + baci_input_file
        + ' '
        + baci_output_file
    )
    return baci_cmd


@pytest.fixture(scope='session')
def baci_post_cmds(baci_job, baci_output_file):

    post_cmds = []
    for id, baci_post_process_option in enumerate(
        baci_job['driver_params']['post_process_options']
    ):
        post_cmd = (
            baci_job['driver_params']['path_to_postprocessor']
            + ' '
            + baci_post_process_option
            + ' --file='
            + baci_output_file
            + ' --output='
            + baci_job['expt_dir']
            + '/'
            + baci_job['expt_name']
            + '_'
            + str(baci_job['id'])
            + '_'
            + str(id + 1)
        )
        post_cmds.append(post_cmd)
    return post_cmds


@pytest.fixture(scope='function')
def ansys_driver(driver_base_settings, mocker):
    """ Generic ANSYS driver"""
    # TODO this is super ugly. creation of DB needs te be moved out of
    # driver init to resolve this
    mocker.patch('pqueens.database.mongodb.MongoDB.__init__', return_value=None)
    driver_base_settings['address'] = 'localhost:27017'
    driver_base_settings['file_prefix'] = 'rst'
    driver_base_settings['output_scratch'] = 'rst'
    my_driver = AnsysDriverNative(None, 'v15', driver_base_settings)
    return my_driver


########################################################################
#########################   DOCKER   ###################################
########################################################################
@pytest.fixture(scope='module')
def baci_docker_job(baci_job, tmpdir_factory):
    """ Job dictionary for testing BACI docker driver"""

    baci_job['driver_params']['docker_container'] = str(
        tmpdir_factory.mktemp('docker_container_dir').join('container')
    )

    return baci_job


@pytest.fixture(scope='module')
def docker_volume_map(job):
    volume_map = {job['expt_dir']: {'bind': job['expt_dir'], 'mode': 'rw'}}
    return volume_map


########################################################################
#########################   DRIVER   ###################################
########################################################################
@pytest.fixture(scope='session')
def driver_base_settings(job):
    """
    A base settings dict that can be used to create Driver object.
    """

    base_settings = dict()

    base_settings['experiment_dir'] = job['expt_dir']
    base_settings['experiment_name'] = job['expt_name']
    base_settings['job_id'] = job['id']
    base_settings['input_file'] = 'input.json'
    base_settings['template'] = 'template.dat'
    base_settings['output_file'] = 'experiment.out'
    base_settings['job'] = job
    base_settings['batch'] = 1
    base_settings['executable'] = 'baci-release'
    base_settings['result'] = 1e-3
    base_settings['postprocessor'] = 'post_drt_mon'
    base_settings['post_options'] = '--field=structure --node=26 --start=1'
    base_settings['postpostprocessor'] = 'Post_post.py'
    base_settings['port'] = 27017
    base_settings['num_procs'] = 4
    base_settings['num_procs_post'] = 2

    return base_settings
