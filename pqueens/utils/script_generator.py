"""Generator module supplies function to generate a job or task script for
submission."""

import os

from pqueens.utils.injector import inject
from pqueens.utils.run_subprocess import run_subprocess


def generate_submission_script(
    script_options, submission_script_path, submission_script_template, connect_to_resource=None
):
    """Generate a submission script for either a task definition on AWS based
    on a task-script template or the simulation based on a job-script template
    (with the latter either local or remote).

    Args:
         job_id (int): Internal QUEENS job-ID that is used to enumerate the simulations
    Returns:
         None
    """
    # local dummy path
    local_dummy_path = os.path.join(os.path.dirname(__file__), 'dummy_submission_script')

    # create actual submission file with parsed parameters
    inject(script_options, submission_script_template, local_dummy_path)

    # copy submission script to specified local or remote location
    if connect_to_resource == None:
        command_list = ['cp', local_dummy_path, submission_script_path]
    else:
        command_list = ['scp', local_dummy_path, connect_to_resource + ':' + submission_script_path]
    command_string = ' '.join(command_list)
    process_returncode, p, stdout, stderr = run_subprocess(command_string)
