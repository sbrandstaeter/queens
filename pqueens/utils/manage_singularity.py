import hashlib
import os

from pqueens.utils.run_subprocess import run_subprocess


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
    def __init__(self, remote_flag, connect_to_resource, cluster_bind, path_to_singularity):
        self.remote_flag = remote_flag
        self.connect_to_resource = connect_to_resource
        self.cluster_bind = cluster_bind
        self.path_to_singularity = path_to_singularity

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
        if self.remote_flag:
            command_list = ['ssh', self.connect_to_resource, '\'echo $SINGULARITY_BIND\'']
        else:
            command_list = ['echo $SINGULARITY_BIND']
        command_string = ' '.join(command_list)
        _, _, stdout, _ = run_subprocess(command_string)
        if stdout == "\n":
            if self.remote_flag:
                command_list = [
                    'ssh',
                    self.connect_to_resource,
                    "\"echo 'export SINGULARITY_BIND="
                    + self.cluster_bind
                    + "\' >> ~/.bashrc && source ~/.bashrc\"",
                ]
            else:
                command_list = [
                    "echo 'export SINGULARITY_BIND="
                    + self.cluster_bind
                    + "\' >> ~/.bashrc && source ~/.bashrc"
                ]
        command_string = ' '.join(command_list)
        _, _, _, _ = run_subprocess(command_string)

        # Create a Singularity PATH variable that is equal to the host PATH
        if self.remote_flag:
            command_list = ['ssh', self.connect_to_resource, '\'echo $SINGULARITYENV_APPEND_PATH\'']
        else:
            command_list = ['echo $SINGULARITYENV_APPEND_PATH']
        command_string = ' '.join(command_list)
        _, _, stdout, _ = run_subprocess(command_string)
        if stdout == "\n":
            if self.remote_flag:
                command_list = [
                    'ssh',
                    self.connect_to_resource,
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
        if self.remote_flag:
            command_list = [
                'ssh',
                self.connect_to_resource,
                '\'echo $SINGULARITYENV_APPEND_LD_LIBRARY_PATH\'',
            ]
        else:
            command_list = ['echo $SINGULARITYENV_APPEND_LD_LIBRARY_PATH']
        command_string = ' '.join(command_list)
        _, _, stdout, _ = run_subprocess(command_string)
        if stdout == "\n":
            if self.remote_flag:
                command_list = [
                    'ssh',
                    self.connect_to_resource,
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
                if self.remote_flag:
                    print("Updating remote image from local image...")
                    print("(This might take a couple of seconds, but needs only to be done once)")
                    rel_path = "../../driver.simg"
                    abs_path = os.path.join(script_dir, rel_path)
                    command_list = [
                        "scp",
                        abs_path,
                        self.connect_to_resource + ':' + self.path_to_singularity,
                    ]
                    command_string = ' '.join(command_list)
                    _, _, _, stderr = run_subprocess(command_string)
                    if stderr:
                        raise RuntimeError(
                            "Error! Was not able to copy local singularity image to remote! "
                            "Abort..."
                        )

            # check existence singularity on remote
            if self.remote_flag:
                command_list = [
                    'ssh -T',
                    self.connect_to_resource,
                    'test -f',
                    self.path_to_singularity + "/driver.simg && echo 'Y' || echo 'N'",
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
                        self.connect_to_resource + ':' + self.path_to_singularity,
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
            if self.remote_flag:
                print("Updating now remote image from local image...")
                print("(This might take a couple of seconds, but needs only to be done once)")
                rel_path = "../../driver.simg"
                abs_path = os.path.join(script_dir, rel_path)
                command_list = [
                    "scp",
                    abs_path,
                    self.connect_to_resource + ':' + self.path_to_singularity,
                ]
                command_string = ' '.join(command_list)
                _, _, _, stderr = run_subprocess(command_string)
                if stderr:
                    raise RuntimeError(
                        "Error! Was not able to copy local singularity image to remote! " "Abort..."
                    )
                print('All singularity images ok! Starting simulation on cluster...')


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
