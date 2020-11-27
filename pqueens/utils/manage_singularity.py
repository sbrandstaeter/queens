import hashlib
import os
import random
import subprocess
import time
from pprint import pprint

from pqueens.utils.run_subprocess import run_subprocess
from pqueens.utils.user_input import request_user_input_with_default_and_timeout


def create_singularity_image():
    """
    Add current QUEENS setup to pre-designed singularity image for cluster applications

    Returns:
         None

    """
    # create the actual image
    command_string = '/usr/bin/singularity --version'
    _, _, _, stderr = run_subprocess(command_string)
    if stderr:
        raise RuntimeError(f'Singularity could not be executed! The error message was: {stderr}')

    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    rel_path1 = '../../driver.simg'
    rel_path2 = '../../singularity_recipe'
    abs_path1 = os.path.join(script_dir, rel_path1)
    abs_path2 = os.path.join(script_dir, rel_path2)
    path_to_pqueens = os.path.join(script_dir, '../')
    command_list = [
        "cd",
        path_to_pqueens,
        "&&",
        "/usr/bin/singularity build --fakeroot",
        abs_path1,
        abs_path2,
    ]
    command_string = ' '.join(command_list)
    returncode, _, stdout, stderr = run_subprocess(command_string)

    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    rel_path = '../../driver.simg'
    abs_path = os.path.join(script_dir, rel_path)
    if returncode or not os.path.isfile(abs_path):
        print('Build of local singularity image failed!')
        print('----------------------------------------------------------------------------------')
        print(f'The returned stdout message was:\n {stdout}\n')
        print(f'The returned error message was:\n {stderr}\n')
        raise RuntimeError(f'The returned error message was: {stderr}, {stdout}')


class SingularityManager:
    def __init__(self, remote, remote_connect, singularity_bind, singularity_path, input_file):
        self.remote = remote
        self.remote_connect = remote_connect
        self.singularity_bind = singularity_bind
        self.singularity_path = singularity_path
        self.input_file = input_file

    def check_singularity_system_vars(self):
        """
        Check and establish necessary system variables for the singularity image.
        Examples are directory bindings such that certain directories of the host can be
        accessed on runtime within the singularity image. Other system variables include
        path and environment variables

        Returns:
            None

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
        _, _, _, _ = run_subprocess(command_string)

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
            _, _, _, _ = run_subprocess(command_string)

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
            _, _, _, _ = run_subprocess(command_string)

    def prepare_singularity_files(self):
        """
        Checks if local and remote singularity images are existent and compares a hash-file
        to the current hash of the files to determine if the singularity image is up to date.
        The method furthermore triggers the build of a new singularity image if necessary.

        Returns:
            None

        """

        # check existence local
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        rel_path = '../../driver.simg'
        abs_path = os.path.join(script_dir, rel_path)
        if os.path.isfile(abs_path):
            # check singularity status local
            command_list = ['/usr/bin/singularity', 'run', abs_path, '--hash=true']
            command_string = ' '.join(command_list)
            _, _, old_data, stderr = run_subprocess(command_string)

            if stderr:
                raise RuntimeError(f'Singularity hash-check return the error: {stderr}. Abort...')

            hashlist = hash_files()
            # Write local singularity image and remote image
            # convert the string that is returned from the singularity image into a list
            old_data = [ele.replace("\'", "") for ele in old_data.strip('][').split(', ')]
            old_data = [ele.replace("]", "") for ele in old_data]
            old_data = [ele.replace("\n", "") for ele in old_data]

            if ''.join(old_data) != ''.join(hashlist):
                print(
                    "Local singularity image is not up-to-date with QUEENS! "
                    "Writing new local image..."
                )
                print("(This will take 3 min or so, but needs only to be done once)")
                # deleting old image
                rel_path = '../../driver*'
                abs_path = os.path.join(script_dir, rel_path)
                command_list = ['rm', abs_path]
                command_string = ' '.join(command_list)
                _, _, _, _ = run_subprocess(command_string)
                create_singularity_image()
                print("Local singularity image written successfully!")

                # Update remote image
                if self.remote:
                    print("Updating remote image from local image...")
                    print("(This might take a couple of seconds, but needs only to be done once)")
                    rel_path = "../../driver.simg"
                    abs_path = os.path.join(script_dir, rel_path)
                    command_list = [
                        "scp",
                        abs_path,
                        self.remote_connect + ':' + self.singularity_path,
                    ]
                    command_string = ' '.join(command_list)
                    _, _, _, stderr = run_subprocess(command_string)
                    if stderr:
                        raise RuntimeError(
                            "Error! Was not able to copy local singularity image to remote! "
                            "Abort..."
                        )

            # check existence singularity on remote
            if self.remote:
                command_list = [
                    'ssh -T',
                    self.remote_connect,
                    'test -f',
                    self.singularity_path + "/driver.simg && echo 'Y' || echo 'N'",
                ]
                command_string = ' '.join(command_list)
                _, _, stdout, _ = run_subprocess(command_string)
                if 'N' in stdout:
                    # Update remote image
                    print(
                        "Remote singularity image is not existent! "
                        "Updating remote image from local image..."
                    )
                    print("(This might take a couple of seconds, but needs only to be done once)")
                    rel_path = "../../driver.simg"
                    abs_path = os.path.join(script_dir, rel_path)
                    command_list = [
                        "scp",
                        abs_path,
                        self.remote_connect + ':' + self.singularity_path,
                    ]
                    command_string = ' '.join(command_list)
                    _, _, _, stderr = run_subprocess(command_string)
                    if stderr:
                        raise RuntimeError(
                            "Error! Was not able to copy local singularity image to remote! "
                            "Abort..."
                        )
                    print('All singularity images ok! Starting simulation on cluster...')

        else:
            # local image was not even existent --> create local and remote image
            print("No local singularity image found! Building new image...")
            print("(This will take 3 min or so, but needs only to be done once)")
            print("_______________________________________________________________________________")
            print("")
            print("Make sure QUEENS was called from the base directory containing the main.py file")
            print("to set the correct relative paths for the image; otherwise abort!")
            print("_______________________________________________________________________________")
            create_singularity_image()
            print("Local singularity image written successfully!")
            if self.remote:
                print("Updating now remote image from local image...")
                print("(This might take a couple of seconds, but needs only to be done once)")
                rel_path = "../../driver.simg"
                abs_path = os.path.join(script_dir, rel_path)
                command_list = [
                    "scp",
                    abs_path,
                    self.remote_connect + ':' + self.singularity_path,
                ]
                command_string = ' '.join(command_list)
                _, _, _, stderr = run_subprocess(command_string)
                if stderr:
                    raise RuntimeError(
                        "Error! Was not able to copy local singularity image to remote! " "Abort..."
                    )
                print('All singularity images ok! Starting simulation on cluster...')

    def kill_previous_queens_ssh_remote(self, username):
        """
        Kill existing ssh-port-forwardings on the remote
        that were caused by previous QUEENS simulations
        that either crashed or are still in place due to other reasons.
        This method will avoid that a user opens too many unnecessary ports on the remote
        and blocks them for other users.

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
            # print the queens related open ports
            print('The following QUEENS sessions are still occupying ports on the remote:')
            print('----------------------------------------------------------------------')
            pprint(active_ssh, width=150)
            print('----------------------------------------------------------------------')
            print('')
            print('Do you want to close these connections (recommended)?')
            while True:
                try:
                    print('Please type "y" or "n" >> ')
                    answer = request_user_input_with_default_and_timeout(default="n", timeout=10)
                except SyntaxError:
                    answer = None

                if answer.lower() == 'y':
                    ssh_ids = [line.split()[1] for line in active_ssh]
                    for ssh_id in ssh_ids:
                        command_list = ['ssh', self.remote_connect, '\'kill -9', ssh_id + '\'']
                        command_string = ' '.join(command_list)
                        _, _, std, err = run_subprocess(command_string)
                    print('Old QUEENS port-forwardings were successfully terminated!')
                    break
                elif answer.lower() == 'n':
                    break
                elif answer is None:
                    print('You gave an empty input! Only "y" or "n" are valid inputs! Try again!')
                else:
                    print(
                        f'The input "{answer}" is not an appropriate choice! '
                        f'Only "y" or "n" are valid inputs!'
                    )
                    print('Try again!')
        else:
            pass

    def establish_port_forwarding_remote(self, address_localhost):
        """
        Automated port-forwarding from localhost to remote machine to forward data to the database
        on localhost's port 27017 and a designated port on the master node of the remote machine

        Args:
            address_localhost (str): IP-address of localhost

        Returns:
            None
        """

        port_fail = 1
        while port_fail != "":
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
            time.sleep(0.1)
        print('Remote port-forwarding successfully established for port %s' % (port))

        return port

    def establish_port_forwarding_local(self, address_localhost):
        """
        Establish a port-forwarding for localhost's port 9001 to the remote's ssh-port 22
        for passwordless communication with the remote machine over ssh

        Args:
            address_localhost (str): IP-address of the localhost

        Returns:
            None

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
        ssh_proc = subprocess.Popen(
            command_list, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stat = ssh_proc.poll()
        while stat is None:
            stat = ssh_proc.poll()
        # TODO Think of some kind of error catching here;
        #  so far it works but error might be cryptical

    def close_remote_port(self, port):
        """
        Closes the ports that were used in the current QUEENS simulation for remote computing.

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
        _, _, active_ssh, _ = run_subprocess(command_string)

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
                _, _, _, _ = run_subprocess(command_string)
            print('Active QUEENS port-forwardings were closed successfully!')

    def copy_temp_json(self):
        """
        Copies a (temporary) JSON input-file on a remote to execute some parts of QUEENS within
        the singularity image on the remote, given the input configurations.

        Returns:
            None

        """

        command_list = [
            "scp",
            self.input_file,
            self.remote_connect + ':' + self.singularity_path + '/temp.json',
        ]
        command_string = ' '.join(command_list)
        _, _, _, stderr = run_subprocess(command_string)
        if stderr:
            raise RuntimeError(
                "Error! Was not able to copy temporary input file to remote! Abort..."
            )

    def copy_post_post(self):
        """
        Copy an instance of the post-post module to the remote and
        load it dynamically during runtime.
        This enables fast changes in post-post scripts without the need to rebuild the singularity
        image.

        Returns:
            None

        """

        abs_dirpath_current_file = os.path.dirname(os.path.abspath(__file__))
        abs_path_post_post = os.path.join(abs_dirpath_current_file, '../post_post')
        # delete old files
        command_list = [
            "ssh",
            self.remote_connect,
            '\'rm -r',
            self.singularity_path + '/post_post\'',
        ]
        command_string = ' '.join(command_list)
        _, _, _, _ = run_subprocess(command_string)

        # copy new files
        command_list = [
            "scp -r",
            abs_path_post_post,
            self.remote_connect + ':' + self.singularity_path + '/post_post',
        ]
        command_string = ' '.join(command_list)
        _, _, _, stderr = run_subprocess(command_string)
        if stderr:
            raise RuntimeError(
                "Error! Was not able to copy post_post directory to remote! Abort..."
            )


def hash_files():
    """
    Hash all files that are used in the singularity image anc check if some files were changed.
    This is important to keep the singularity image always up to date with the code base

    Returns:
        None

    """
    hashlist = []
    hasher = hashlib.md5()
    # hash all drivers
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    rel_path = "../drivers"
    abs_path = os.path.join(script_dir, rel_path)
    elements = os.listdir(abs_path)
    elements.sort()
    filenames = [
        os.path.join(abs_path, ele) for _, ele in enumerate(elements) if ele.endswith('.py')
    ]
    for filename in filenames:
        with open(filename, 'rb') as inputfile:
            data = inputfile.read()
            hasher.update(data)
        hashlist.append(hasher.hexdigest())

    # hash mongodb
    rel_path = "../database/mongodb.py"
    abs_path = os.path.join(script_dir, rel_path)
    with open(abs_path, 'rb') as inputfile:
        data = inputfile.read()
        hasher.update(data)
    hashlist.append(hasher.hexdigest())

    # hash utils
    rel_path = '../utils/injector.py'
    abs_path = os.path.join(script_dir, rel_path)
    with open(abs_path, 'rb') as inputfile:
        data = inputfile.read()
        hasher.update(data)
    hashlist.append(hasher.hexdigest())

    rel_path = '../utils/run_subprocess.py'
    abs_path = os.path.join(script_dir, rel_path)
    with open(abs_path, 'rb') as inputfile:
        data = inputfile.read()
        hasher.update(data)
    hashlist.append(hasher.hexdigest())

    # hash setup_remote
    rel_path = '../../setup_remote.py'
    abs_path = os.path.join(script_dir, rel_path)
    with open(abs_path, 'rb') as inputfile:
        data = inputfile.read()
        hasher.update(data)
    hashlist.append(hasher.hexdigest())

    # hash remote_main
    rel_path = '../remote_main.py'
    abs_path = os.path.join(script_dir, rel_path)
    with open(abs_path, 'rb') as inputfile:
        data = inputfile.read()
        hasher.update(data)
    hashlist.append(hasher.hexdigest())

    # hash postpost files
    rel_path = '../post_post/post_post.py'
    abs_path = os.path.join(script_dir, rel_path)
    with open(abs_path, 'rb') as inputfile:
        data = inputfile.read()
        hasher.update(data)
    hashlist.append(hasher.hexdigest())

    return hashlist
