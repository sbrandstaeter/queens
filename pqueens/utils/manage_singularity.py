"""Singularity management utilities."""
import logging
import os
import random
import subprocess
import time

from pqueens.utils.path_utils import (
    PATH_TO_QUEENS,
    relative_path_from_pqueens,
    relative_path_from_queens,
)
from pqueens.utils.run_subprocess import SubprocessError, run_subprocess
from pqueens.utils.user_input import request_user_input_with_default_and_timeout

_logger = logging.getLogger(__name__)
ABS_SINGULARITY_IMAGE_PATH = relative_path_from_queens("singularity_image.sif")


def create_singularity_image():
    """Create pre-designed singularity image for cluster applications."""
    # create the actual image
    command_string = 'singularity --version'
    run_subprocess(command_string, additional_error_message='Singularity could not be executed!')

    definition_path = 'singularity_recipe.def'
    abs_definition_path = relative_path_from_queens(definition_path)
    command_list = [
        "cd",
        str(PATH_TO_QUEENS),
        "&& unset SINGULARITY_BIND &&",
        "singularity build --force --fakeroot",
        ABS_SINGULARITY_IMAGE_PATH,
        abs_definition_path,
    ]
    command_string = ' '.join(command_list)

    # Singularity logs to the wrong stream depending on the OS.
    try:
        run_subprocess(
            command_string, additional_error_message='Build of local singularity image failed!'
        )
    except SubprocessError as sp_error:
        # Check if build was successful
        if str(sp_error).find("INFO:    Build complete:") < 0:
            raise sp_error

    if not os.path.isfile(ABS_SINGULARITY_IMAGE_PATH):
        raise FileNotFoundError(f'No singularity image "{ABS_SINGULARITY_IMAGE_PATH}" found')


class SingularityManager:
    """Singularity management class."""

    def __init__(self, remote, remote_connect, singularity_bind, singularity_path, input_file):
        """Init method for the singularity object.

        Args:
            remote (bool): True if the simulation runs are remote
            remote_connect (str): String of user@remote_machine
            singularity_bind (str): Binds for the singularity runs
            singularity_path (path): Path to singularity exec
            input_file (path): Path to QUEENS input file
        """
        self.remote = remote
        self.remote_connect = remote_connect
        self.singularity_bind = singularity_bind
        self.singularity_path = singularity_path
        self.input_file = input_file

        if self.remote and self.remote_connect is None:
            raise ValueError(
                "Remote singularity option is set to true but no remote connect is supplied."
            )

    def check_singularity_system_vars(self):
        """Check and establish system variables for the singularity image.

        Examples are directory bindings such that certain directories of
        the host can be accessed on runtime within the singularity
        image. Other system variables include path and environment
        variables.
        """
        # Check if SINGULARITY_BIND exists and if not write it to .bashrc file
        if self.remote:
            command_list = ['ssh', self.remote_connect, '\'echo $SINGULARITY_BIND\'']
        else:
            command_list = ['echo $SINGULARITY_BIND']
        command_string = ' '.join(command_list)
        _, _, stdout, _ = run_subprocess(command_string)
        if stdout == "\n":
            if self.remote:
                command_list = [
                    'ssh',
                    self.remote_connect,
                    "\"echo 'export SINGULARITY_BIND="
                    + self.singularity_bind
                    + "\' >> ~/.bashrc && source ~/.bashrc\"",
                ]
            else:
                command_list = [
                    "echo 'export SINGULARITY_BIND="
                    + self.singularity_bind
                    + "\' >> ~/.bashrc && source ~/.bashrc"
                ]
        command_string = ' '.join(command_list)
        run_subprocess(command_string)

        # Create a Singularity PATH variable that is equal to the host PATH
        if self.remote:
            command_list = ['ssh', self.remote_connect, '\'echo $SINGULARITYENV_APPEND_PATH\'']
        else:
            command_list = ['echo $SINGULARITYENV_APPEND_PATH']
        command_string = ' '.join(command_list)
        _, _, stdout, _ = run_subprocess(command_string)
        if stdout == "\n":
            if self.remote:
                command_list = [
                    'ssh',
                    self.remote_connect,
                    # pylint: disable=line-too-long
                    "\"echo 'export SINGULARITYENV_APPEND_PATH=$PATH' >> ~/.bashrc && source "
                    "~/.bashrc\"",
                    # pylint: enable=line-too-long
                ]  # noqa
            else:
                command_list = [
                    # pylint: disable=line-too-long
                    "echo 'export SINGULARITYENV_APPEND_PATH=$PATH' >> ~/.bashrc && source "
                    "~/.bashrc"
                    # pylint: enable=line-too-long
                ]  # noqa
            command_string = ' '.join(command_list)
            run_subprocess(command_string)

        # Create a Singulartity LD_LIBRARY_PATH variable that is equal to the host
        # LD_LIBRARY_PATH
        if self.remote:
            command_list = [
                'ssh',
                self.remote_connect,
                '\'echo $SINGULARITYENV_APPEND_LD_LIBRARY_PATH\'',
            ]
        else:
            command_list = ['echo $SINGULARITYENV_APPEND_LD_LIBRARY_PATH']
        command_string = ' '.join(command_list)
        _, _, stdout, _ = run_subprocess(command_string)
        if stdout == "\n":
            if self.remote:
                command_list = [
                    'ssh',
                    self.remote_connect,
                    # pylint: disable=line-too-long
                    "\"echo 'export SINGULARITYENV_APPEND_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' >> "
                    "~/.bashrc && source ~/.bashrc\"",
                    # pylint: enable=line-too-long
                ]  # noqa
            else:
                command_list = [
                    # pylint: disable=line-too-long
                    "echo 'export SINGULARITYENV_APPEND_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' >> "
                    "~/.bashrc && source ~/.bashrc"
                    # pylint: enable=line-too-long
                ]  # noqa
            command_string = ' '.join(command_list)
            run_subprocess(command_string)

    def copy_image_to_remote(self):
        """Copy the local singularity image to the remote resource."""
        _logger.info("Updating remote image from local image...")
        _logger.info("(This might take a couple of seconds, but needs only to be done once)")
        command_list = [
            "scp",
            ABS_SINGULARITY_IMAGE_PATH,
            self.remote_connect + ':' + str(self.singularity_path),
        ]
        command_string = ' '.join(command_list)
        run_subprocess(
            command_string,
            additional_error_message="Was not able to copy local singularity image to remote! ",
        )

    def prepare_singularity_files(self):
        """Checks if local and remote singularity images are existent.

        Compares a hash-file to the current hash of the files to determine if
        the singularity image is up-to-date. The method furthermore triggers
        the build of a new singularity image if necessary.

        Returns:
            None
        """
        copy_to_remote = False
        if _check_if_new_image_needed():
            _logger.info(
                "Local singularity image is not up-to-date with QUEENS! "
                "Writing new local image..."
            )
            _logger.info("(This will take 3 min or so, but needs only to be done once)")
            create_singularity_image()
            _logger.info("Local singularity image written successfully!")

            if self.remote:
                copy_to_remote = True
        else:
            _logger.info("Found an up-to-date local singularity image.")

        if self.remote and not copy_to_remote:
            try:
                remote_hash = sha1sum(
                    str(self.singularity_path.joinpath('singularity_image.sif')),
                    self.remote_connect,
                )
                local_hash = sha1sum(ABS_SINGULARITY_IMAGE_PATH)
                copy_to_remote = remote_hash != local_hash
            except:
                copy_to_remote = True

        if copy_to_remote:
            self.copy_image_to_remote()

    def kill_previous_queens_ssh_remote(self, username):
        """Kill existing ssh-port-forwardings on the remote machine.

        These were caused by previous QUEENS simulations that either crashed or are still in place
        due to other reasons. This method will avoid that a user opens too many unnecessary ports
        on the remote and blocks them for other users.

        Args:
            username (string): Username of person logged in on remote machine

        Returns:
            None
        """
        # find active queens ssh ports on remote
        command_list = [
            'ssh',
            self.remote_connect,
            '\'ps -aux | grep ssh | grep',
            username.rstrip(),
            '| grep :localhost:27017\'',
        ]

        command_string = ' '.join(command_list)
        _, _, active_ssh, _ = run_subprocess(command_string)

        # skip entries that contain "grep" as this is the current command
        try:
            active_ssh = [line for line in active_ssh.splitlines() if not 'grep' in line]
        except IndexError:
            pass

        if active_ssh:
            # _logger.info the queens related open ports
            _logger.info('The following QUEENS sessions are still occupying ports on the remote:')
            _logger.info('----------------------------------------------------------------------')
            for line in active_ssh:
                _logger.info(line)
            _logger.info('----------------------------------------------------------------------')
            _logger.info('')
            _logger.info('Do you want to close these connections (recommended)?')
            while True:
                try:
                    _logger.info('Please type "y" or "n" >> ')
                    answer = request_user_input_with_default_and_timeout(default="n", timeout=10)
                except SyntaxError:
                    answer = None

                if answer.lower() == 'y':
                    ssh_ids = [line.split()[1] for line in active_ssh]
                    for ssh_id in ssh_ids:
                        command_list = ['ssh', self.remote_connect, '\'kill -9', ssh_id + '\'']
                        command_string = ' '.join(command_list)
                        run_subprocess(command_string)
                    _logger.info('Old QUEENS port-forwardings were successfully terminated!')
                    break

                if answer.lower() == 'n':
                    break
                if answer is None:
                    _logger.info(
                        'You gave an empty input! Only "y" or "n" are valid inputs! Try again!'
                    )
                else:
                    _logger.info(
                        'The input %s is not an appropriate choice! '
                        'Only "y" or "n" are valid inputs!',
                        answer,
                    )
                    _logger.info('Try again!')
        else:
            pass

    def establish_port_forwarding_remote(self, address_localhost):
        """Automated port-forwarding from localhost to remote machine.

        Forward data to the database on localhost's port 27017 and a designated
        port on the master node of the remote machine.

        Args:
            address_localhost (str): IP-address of localhost

        Returns:
            None
        """
        _logger.info('Establish remote port-forwarding')
        port_fail = 1
        max_attempts = 100
        attempts = 1
        while port_fail != "" and attempts < max_attempts:
            port = random.randrange(2030, 20000, 1)
            command_list = [
                'ssh',
                '-t',
                '-t',
                self.remote_connect,
                '\'ssh',
                '-fN',
                '-g',
                '-L',
                str(port) + r':' + 'localhost' + r':27017',
                address_localhost + '\'',
            ]
            command_string = ' '.join(command_list)
            port_fail = os.popen(command_string).read()
            _logger.info(f'attempt #{attempts}: {command_string}')
            _logger.debug('which returned: {port_fail}')
            time.sleep(0.1)
        _logger.info('Remote port-forwarding established successfully for port %s', port)

        return port

    def establish_port_forwarding_local(self, address_localhost):
        """Establish port-forwarding from local to remote.

        Establish a port-forwarding for localhost's port 9001 to the
        remote's ssh-port 22 for passwordless communication with the remote
        machine over ssh.

        Args:
            address_localhost (str): IP-address of the localhost
        """
        remote_address = self.remote_connect.split(r'@')[1]
        command_list = [
            'ssh',
            '-f',
            '-N',
            '-L',
            r'9001:' + remote_address + r':22',
            address_localhost,
        ]
        with subprocess.Popen(
            command_list, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as ssh_proc:
            stat = ssh_proc.poll()
            while stat is None:
                stat = ssh_proc.poll()
            # Think of some kind of error catching here;
            # so far it works but error might be cryptical

    def close_local_port_forwarding(self):
        """Closes port forwarding from local to remote machine."""
        _, _, username, _ = run_subprocess('whoami')
        command_string = "ps -aux | grep 'ssh -f -N -L 9001:' | grep ':22 " + username + "@'"
        _, _, active_ssh, _ = run_subprocess(
            command_string, raise_error_on_subprocess_failure=False
        )

        if active_ssh:
            active_ssh_ids = []
            try:
                active_ssh_ids = [
                    line.split()[1] for line in active_ssh.splitlines() if not 'grep' in line
                ]
            except IndexError:
                pass

            if active_ssh_ids:
                for ssh_id in active_ssh_ids:
                    command_string = 'kill -9 ' + ssh_id
                    run_subprocess(command_string, raise_error_on_subprocess_failure=False)
                _logger.info(
                    'Active QUEENS local to remote port-forwardings were closed successfully!'
                )

    def close_remote_port(self, port):
        """Closes the ports used in the current QUEENS simulation.

        Args:
            port (int): Random port selected previously
        Returns:
            None
        """
        # get the process id of open port
        _, _, username, _ = run_subprocess('whoami')
        command_list = [
            'ssh',
            self.remote_connect,
            '\'ps -aux | grep ssh | grep',
            username.rstrip(),
            '| grep',
            str(port) + ':localhost:27017\'',
        ]
        command_string = ' '.join(command_list)
        _, _, active_ssh, _ = run_subprocess(
            command_string, raise_error_on_subprocess_failure=False
        )

        # skip entries that contain "grep" as this is the current command
        try:
            active_ssh_ids = [
                line.split()[1] for line in active_ssh.splitlines() if not 'grep' in line
            ]
        except IndexError:
            pass

        if active_ssh_ids != '':
            for ssh_id in active_ssh_ids:
                command_list = ['ssh', self.remote_connect, '\'kill -9', ssh_id + '\'']
                command_string = ' '.join(command_list)
                run_subprocess(command_string)
            _logger.info('Active QUEENS remote to local port-forwardings were closed successfully!')

    def copy_temp_json(self):
        """Copies a (temporary) JSON input-file to the remote machine.

        Is needed to execute some parts of QUEENS within the singularity image on the remote,
        given the input configurations.

        Returns:
            None
        """
        command_list = [
            "scp",
            str(self.input_file),
            self.remote_connect + ':' + str(self.singularity_path.joinpath('temp.json')),
        ]
        command_string = ' '.join(command_list)
        run_subprocess(
            command_string,
            additional_error_message="Was not able to copy temporary input file to remote!",
        )


def _check_if_new_image_needed():
    """Indicate if a new singularity image needs to be build.

    Before checking if the files changed, a check is performed to see if there is an image first.

    Returns:
        (bool): True if new image is needed.
    """
    if os.path.exists(ABS_SINGULARITY_IMAGE_PATH):
        return _check_if_files_changed()
    return True


def _check_if_files_changed():
    """Indicates if the source files deviate w.r.t. to singularity container.

    Returns:
        [bool]: if files have changed
    """
    # Folders included in the singularity image relevant for a run
    folders_to_compare_list = [
        'drivers/',
        'data_processor/',
        'utils/',
        'external_geometry/',
        'randomfields/',
    ]

    # Specific files in the singularity image relevant for a run
    files_to_compare_list = [
        "database/mongodb.py",
        '../setup_remote.py',
        'remote_main.py',
    ]
    # generate absolute paths
    files_to_compare_list = [relative_path_from_pqueens(file) for file in files_to_compare_list]
    folders_to_compare_list = [relative_path_from_pqueens(file) for file in folders_to_compare_list]

    # Add files from the relevant folders to the list of files
    for folder in folders_to_compare_list:
        files_to_compare_list.extend(_get_python_files_in_folder(folder))

    files_changed = False
    for file in files_to_compare_list:
        # File path inside the container
        filepath_in_singularity = '/queens/pqueens/' + file.split("queens/pqueens/")[-1]

        # Compare the queens source files with the ones inside the container
        command_string = (
            f"singularity exec {ABS_SINGULARITY_IMAGE_PATH} "
            + f"cmp {file} {filepath_in_singularity}"
        )
        _, _, stdout, stderr = run_subprocess(
            command_string, raise_error_on_subprocess_failure=False
        )

        # If file is different or missing stop iteration and build the image
        if stdout or stderr:
            files_changed = True
            break
    return files_changed


def _get_python_files_in_folder(relative_path):
    """Get list of absolute paths of files in folder.

    Only python files are included.

    Args:
        relative_path (str): Relative path to folder from pqueens.

    Returns:
        file_paths: List of the absolute paths of the python files within the folder.
    """
    abs_path = relative_path_from_pqueens(relative_path)
    elements = os.listdir(abs_path)
    elements.sort()
    file_paths = [
        os.path.join(abs_path, ele) for _, ele in enumerate(elements) if ele.endswith('.py')
    ]
    return file_paths


def sha1sum(file_path, remote_connect=None):
    """Hash files using sha1sum.

    sha1sum is a computer program that calculates hashes and is the default on most Linux
    distributions. It it is not available on your OS under the same name you can still create a
    symlink.
    Args:
        file_path (str): Absolute path to the file to hash.
        remote_connect (str, optional): username@machine in case of remote machines.

    Returns:
        str: the hash of the file
    """
    command_string = f"sha1sum {file_path}"
    if remote_connect:
        command_string = f"ssh {remote_connect} && " + command_string
    _, _, output, _ = run_subprocess(
        command_string,
        additional_error_message="Was not able to hash the file",
    )
    return output.split(" ")[0]
