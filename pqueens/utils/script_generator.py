"""Function to generate a job or task script for submission."""
import shutil

from pqueens.utils.config_directories import LOCAL_TEMPORARY_SUBMISSION_SCRIPT
from pqueens.utils.injector import inject
from pqueens.utils.run_subprocess import run_subprocess


def generate_submission_script(
    script_options, submission_script_path, submission_script_template, user_at_host=None
):
    """Generate a submission script.

    Based on a job-script template.

    Args:
        script_options (dict): Options for the submission
        submission_script_path (Path): Destination path for the script
        submission_script_template (Path): Path to submission template
        user_at_host (str, optional): Resource connection string in the form of username@hostname
    """
    # create actual submission file with parsed parameters
    inject(script_options, submission_script_template, LOCAL_TEMPORARY_SUBMISSION_SCRIPT)

    # copy submission script to specified local or remote location
    if user_at_host is None:
        shutil.copy(LOCAL_TEMPORARY_SUBMISSION_SCRIPT, submission_script_path)
    else:
        command_list = [
            'scp',
            str(LOCAL_TEMPORARY_SUBMISSION_SCRIPT),
            user_at_host + ':' + str(submission_script_path),
        ]
        command_string = ' '.join(command_list)
        run_subprocess(command_string)
