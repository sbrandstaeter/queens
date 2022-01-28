"""Module supplies functions to conduct operation on remote resource."""

from pqueens.utils.run_subprocess import run_subprocess


def make_directory_on_remote(remote_connect, directory):
    """Make (empty) directory on remote resource."""
    command_list = [
        'ssh',
        remote_connect,
        '"mkdir -p',
        directory,
        '"',
    ]
    command_string = ' '.join(command_list)
    run_subprocess(
        command_string, additional_error_message="Directory could not be made on remote machine!"
    )


def copy_directory_to_remote(remote_connect, local_dir, remote_dir):
    """Copy (local) directory to remote resource."""
    command_list = [
        "scp -r ",
        local_dir,
        " ",
        remote_connect,
        ":",
        remote_dir,
    ]
    command_string = ' '.join(command_list)
    run_subprocess(
        command_string, additional_error_message="Directory could not be copied to remote machine!"
    )
