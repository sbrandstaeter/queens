"""Fixtures needed for testing the Driver classes."""
import pytest


@pytest.fixture(scope='session')
def job(tmpdir_factory):
    """Generic job dictionary for testing drivers."""
    job_dict = {}
    job_dict['experiment_dir'] = str(tmpdir_factory.mktemp('experiment_dir'))
    job_dict['experiment_name'] = 'experiment_name'
    job_dict['id'] = 666
    job_dict['params'] = {'alpha': 3.14, 'beta': 2.3}
    job_dict['status'] = 'unknown'

    return job_dict


@pytest.fixture(scope='session')
def baci_job(job, tmpdir_factory):
    """Generic job dictionary for testing Baci."""
    baci_dir = tmpdir_factory.mktemp('baci_dir')

    job['simulation_input_template'] = str(
        tmpdir_factory.mktemp('template_dir').join('template.dat')
    )
    job['path_to_executable'] = str(baci_dir.join('baci_release'))
    job['path_to_postprocessor'] = str(baci_dir.join('post_processor'))
    job['post_process_options'] = ['post_process_options_1', 'post_process_options_2']
    job['data_processor_script'] = str(
        tmpdir_factory.mktemp('data_processor_dir').join('data_processor_script.py')
    )
    return job


@pytest.fixture(scope='session')
def baci_input_file(job):
    """BACI input file created by inject based on job description."""
    baci_input_file = (
        job['experiment_dir'] + '/' + job['experiment_name'] + '_' + str(job['id']) + '.dat'
    )
    return baci_input_file


@pytest.fixture(scope='session')
def baci_output_file(job):
    """BACI output file based on job description."""
    baci_output_file = job['experiment_dir'] + '/' + job['experiment_name'] + '_' + str(job['id'])
    return baci_output_file


@pytest.fixture(scope='session')
def baci_cmd(baci_job, baci_input_file, baci_output_file):
    """Shell command to execute BACI."""
    baci_cmd = baci_job['path_to_executable'] + ' ' + baci_input_file + ' ' + baci_output_file
    return baci_cmd


@pytest.fixture(scope='session')
def baci_post_cmds(baci_job, baci_output_file):
    """Shell command to post process BACI simulation."""
    post_cmds = []
    for id, baci_post_process_option in enumerate(baci_job['post_process_options']):
        post_cmd = (
            baci_job['path_to_postprocessor']
            + ' '
            + baci_post_process_option
            + ' --file='
            + baci_output_file
            + ' --output='
            + baci_job['experiment_dir']
            + '/'
            + baci_job['experiment_name']
            + '_'
            + str(baci_job['id'])
            + '_'
            + str(id + 1)
        )
        post_cmds.append(post_cmd)
    return post_cmds


########################################################################
#########################   DRIVER   ###################################
########################################################################
@pytest.fixture(scope='session')
def driver_base_settings(job):
    """A base settings dict that can be used to create Driver object."""
    base_settings = {}

    base_settings['driver_name'] = 'my_driver'
    base_settings['experiment_name'] = job['experiment_name']
    base_settings['global_output_dir'] = job['experiment_dir']
    base_settings['experiment_dir'] = job['experiment_dir']
    base_settings['scheduler_type'] = 'standard'
    base_settings['remote'] = False
    base_settings['remote_connect'] = None
    base_settings['remote_python_cmd'] = None
    base_settings['singularity'] = False
    base_settings['docker_image'] = None
    base_settings['num_procs'] = 4
    base_settings['num_procs_post'] = 2
    base_settings['cluster_options'] = None
    base_settings['job_id'] = job['id']
    base_settings['simulation_input_template'] = 'template.dat'
    base_settings['job'] = job
    base_settings['batch'] = 1
    base_settings['executable'] = 'baci-release'
    base_settings['result'] = 1e-3
    base_settings['port'] = 27017
    base_settings['do_postprocessing'] = False
    base_settings['postprocessor'] = 'post_drt_mon'
    base_settings['post_options'] = '--field=structure --node=26 --start=1'
    base_settings['do_data_processing'] = True
    base_settings['data_processor'] = 'data_processor.py'
    base_settings['cae_output_streaming'] = False
    base_settings['input_file'] = 'input.json'
    base_settings['input_file_2'] = None
    base_settings['case_run_script'] = None
    base_settings['output_prefix'] = None
    base_settings['output_directory'] = None
    base_settings['local_output_dir'] = None
    base_settings['output_file'] = 'experiment.out'
    base_settings['control_file'] = None
    base_settings['log_file'] = None
    base_settings['error_file'] = None

    return base_settings
