"""Module supplies functions to conduct operation on remote resource."""

import pickle
import uuid
from functools import partial
from pathlib import Path

import cloudpickle
from fabric import Connection

from pqueens.utils.run_subprocess import run_subprocess


def make_directory_on_remote(remote_connect, directory):
    """Make (empty) directory on remote resource.

    Args:
        remote_connect: TODO_doc
        directory: TODO_doc
    """
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
    """Copy (local) directory to remote resource.

    Args:
        remote_connect: TODO_doc
        local_dir: TODO_doc
        remote_dir: TODO_doc
    """
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


class RemoteConnection(Connection):
    """This is class wrapper around fabric.Connection."""

    def __init__(self, host, remote_python, user=None):
        super().__init__(host, user=user)
        self.func_file_name = f"temp_func_{str(uuid.uuid4())}.pickle"
        self.output_file_name = f"output_{str(uuid.uuid4())}.pickle"
        self.python_cmd = (
            f"{remote_python} -c 'import pickle; from pathlib import Path;"
            f"file = open(\"{self.func_file_name}\", \"rb\");"
            f"func = pickle.load(file); file.close();"
            f"Path(\"{self.func_file_name}\").unlink(); result = func();"
            f"file = open(\"{self.output_file_name}\", \"wb\");"
            f"pickle.dump(result, file); file.close();'"
        )

    def run_function(self, func, *func_args, asynchronously=False, **func_kwargs):
        """Run a python function remotely using a ssh connection."""
        partial_func = partial(func, *func_args, **func_kwargs)  # insert function arguments
        with open(self.func_file_name, "wb") as file:
            cloudpickle.dump(partial_func, file)  # pickle function by value

        self.put(self.func_file_name)  # upload local function file
        Path(self.func_file_name).unlink()  # delete local function file

        if asynchronously:
            return self.client.exec_command(self.python_cmd, get_pty=True)

        self.run(self.python_cmd)  # run function remote
        self.get(self.output_file_name)  # download result

        self.run(f'rm {self.output_file_name}')  # delete remote files

        with open(self.output_file_name, 'rb') as file:  # read return value from output file
            return_value = pickle.load(file)

        Path(self.output_file_name).unlink()  # delete local output file

        return return_value
