"""Function to generate a job or task script for submission."""

from pqueens.utils.injector import inject
from pqueens.utils.path_utils import relative_path_from_pqueens
from pqueens.utils.run_subprocess import run_subprocess


def generate_submission_script(
    script_options, submission_script_path, submission_script_template, connect_to_resource=None
):
    """Generate a submission script.

    Based on a job-script template.

    Args:
        script_options (dict): Options for the submission
        submission_script_path (str): Destination path for the script
        submission_script_template (str): Path to submission template
        connect_to_resource (str, optional): Resource connection string
    """
    # local dummy path
    local_dummy_path = relative_path_from_pqueens('utils/dummy_submission_script')

    # create actual submission file with parsed parameters
    inject(script_options, submission_script_template, local_dummy_path)

    # copy submission script to specified local or remote location
    if connect_to_resource is None:
        command_list = ['cp', local_dummy_path, submission_script_path]
    else:
        command_list = ['scp', local_dummy_path, connect_to_resource + ':' + submission_script_path]
    command_string = ' '.join(command_list)

    run_subprocess(command_string)
