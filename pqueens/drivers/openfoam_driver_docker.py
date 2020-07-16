""" This should be a docstring """

import os
import stat
import shutil
import docker
import getpass
from pqueens.drivers.driver import Driver
from pqueens.utils.injector import inject


class OpenFOAMDriverDocker(Driver):
    """ Driver to run OpenFOAM in Docker container

        Attributes:

    """

    def __init__(self, base_settings):
        # TODO dunder init should not be called with dict
        self.docker_image = base_settings['docker_image']
        super(OpenFOAMDriverDocker, self).__init__(base_settings)

    @classmethod
    def from_config_create_driver(cls, config, base_settings, workdir=None):
        """ Create Driver from input file

        Args:
            config (dict):          Input options
            base_settings (dict):   Second dict with input options TODO should probably be removed

        Returns:
            driver: OpenFOAMDriverDocker object
        """
        base_settings['address'] = 'localhost:27017'
        base_settings['docker_image'] = config['driver']['driver_params']['docker_image']
        return cls(base_settings)

    def setup_dirs_and_files(self):
        """ Setup directory structure """

        # extract name of Docker image and potential sudo
        docker_image_list = self.docker_image.split()
        self.image_name = docker_image_list[0]
        if (len(docker_image_list)) == 2:
            self.sudo = docker_image_list[1]
        else:
            self.sudo = ''

        # extract name of input directory and two input dictionaries from input_template string
        input_dir, self.input_dic_1, self.input_dic_2 = self.simulation_input_template.split()

        # define destination directory
        dest_dir = os.path.join(str(self.experiment_dir), str(self.job_id))

        # copy OpenFOAM case directory to output directory in destination directory
        self.case_dir = os.path.join(dest_dir, "output")
        if not os.path.isdir(self.case_dir):
            shutil.copytree(input_dir, self.case_dir)

        # copy OpenFOAM general run script to OpenFOAM case directory as
        # executable/writable case run script, with current case directory
        # inserted
        inject_params = {'case_directory': self.case_dir}
        self.case_run_script = os.path.join(self.case_dir, "run_script")
        inject(inject_params, self.executable, self.case_run_script)
        os.chmod(self.case_run_script, (stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR))

    def run_job(self):
        """ Actual method to run the job on computing machine
            using run_subprocess method from base class
        """
        # first alternative (used currently):
        # explicitly assemble run command for Docker container
        command_string = self.assemble_command_string()

        # run OpenFOAM in Docker container via subprocess
        _, stderr, self.pid = self.run_subprocess(command_string)

        # second alternative (not used currently): use Docker SDK
        # get Docker client
        # client = docker.from_env()

        # assemble environment for Docker container
        # env = {"USER": getpass.getuser(),
        #       "QT_X11_NO_MITSHM": "1",
        #       "QT_XKB_CONFIG_ROOT": "/usr/share/X11/xkb"}

        # assemble volume map for docker container
        # volume_map = {self.experiment_dir: {'bind': self.experiment_dir, 'mode': 'rw'},
        #              '/etc/group': {'bind': '/etc/group', 'mode': 'ro'},
        #              '/etc/passwd': {'bind': '/etc/passwd', 'mode': 'ro'},
        #              '/etc/shadow': {'bind': '/etc/shadow', 'mode': 'ro'},
        #              '/etc/sudoers.d': {'bind': '/etc/sudoers.d', 'mode': 'ro'},
        #              '/tmp/.X11-unix': {'bind': '/tmp/.X11-unix', 'mode': 'ro'}}

        # run OpenFOAM in Docker container
        # stderr = client.containers.run(self.image_name,
        #                               self.case_run_script,
        #                               remove=True,
        #                               user=os.geteuid(),
        #                               environment=env,
        #                               working_dir= os.environ['HOME'],
        #                               volumes=volume_map,
        #                               stdout=False,
        #                               stderr=True)

        # detection of failed jobs
        if stderr:
            self.result = None
            self.job['status'] = 'failed'

    def assemble_command_string(self):
        """  Assemble command list

            Returns:
                list: command list to execute OpenFoam in Docker container

        """
        command_list = [
            self.sudo,
            " docker run -i --rm --user='",
            str(os.geteuid()),
            "' -e USER='",
            getpass.getuser(),
            "' -e QT_X11_NO_MITSHM=1 ",
            "-e QT_XKB_CONFIG_ROOT=/usr/share/X11/xkb ",
            "--workdir='",
            self.experiment_dir,
            "' --volume='",
            self.experiment_dir,
            ":",
            self.experiment_dir,
            "' ",
            "--volume='/etc/group:/etc/group:ro' ",
            "--volume='/etc/passwd:/etc/passwd:ro' ",
            "--volume='/etc/shadow:/etc/shadow:ro' ",
            "--volume='/etc/sudoers.d:/etc/sudoers.d:ro' ",
            "-v=/tmp/.X11-unix:/tmp/.X11-unix ",
            self.image_name,
            " ",
            self.case_run_script,
        ]

        return ''.join(filter(None, command_list))
