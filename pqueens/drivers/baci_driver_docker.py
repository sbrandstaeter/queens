import os
import docker
from pqueens.drivers.driver import Driver
from pqueens.utils.run_subprocess import run_subprocess


class BaciDriverDocker(Driver):
    """
    Driver to run BACI in Docker container

    Attributes:
        docker_image (str): Path to the docker image

    """

    def __init__(self, base_settings):
        # TODO dunder init should not be called with dict
        self.docker_image = base_settings['docker_image']
        super(BaciDriverDocker, self).__init__(base_settings)

    @classmethod
    def from_config_create_driver(cls, config, base_settings):
        """ Create Driver from input file

        Args:
            config (dict):          Input options
            base_settings (dict):   Second dict with input options TODO should probably be removed

        Returns:
            driver (obj): BaciDriverDocker object

        """
        base_settings['address'] = 'localhost:27017'
        base_settings['docker_image'] = config['driver']['driver_params']['docker_image']
        return cls(base_settings)

    def setup_dirs_and_files(self):
        """ Setup directory structure

            Args:
                driver_options (dict): Options dictionary

            Returns:
                None

        """
        # extract name of Docker image and potential sudo
        docker_image_list = self.docker_image.split()
        self.image_name = docker_image_list[0]
        if (len(docker_image_list)) == 2:
            self.sudo = docker_image_list[1]
        else:
            self.sudo = ''

        # define destination directory
        dest_dir = os.path.join(str(self.experiment_dir), str(self.job_id))

        self.output_directory = os.path.join(dest_dir, "output")
        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory)

        # create input file name
        input_string = str(self.experiment_name) + '_' + str(self.job_id) + '.dat'
        self.input_file = os.path.join(dest_dir, input_string)

        # create output file name
        output_string = str(self.experiment_name) + '_' + str(self.job_id)
        self.output_file = os.path.join(self.output_directory, output_string)

    def run_job(self):
        """ Actual method to run the job on computing machine
            using run_subprocess method from base class
        """
        # assemble BACI run command sttring
        self.baci_run_command_string = self.assemble_baci_run_command_string()

        # first alternative (used currently):
        # explicitly assemble run command for Docker container
        docker_run_command_string = self.assemble_docker_run_command_string()

        # run BACI in Docker container via subprocess
        returncode, self.pid = run_subprocess(docker_run_command_string)

        # second alternative (not used currently): use Docker SDK
        # get Docker client
        # client = docker.from_env()

        # assemble volume map for docker container
        # volume_map = {self.experiment_dir: {'bind': self.experiment_dir, 'mode': 'rw'}}

        # run BACI in Docker container via SDK
        # stderr = client.containers.run(self.image_name,
        #                               self.baci_run_command_string,
        #                               remove=True,
        #                               volumes=volume_map,
        #                               stdout=False,
        #                               stderr=True)

        # detection of failed jobs
        if returncode:
            self.result = None
            self.job['status'] = 'failed'

    def assemble_baci_run_command_string(self):
        """  Assemble BACI run command list

            Returns:
                list: command list to execute BACI

        """
        # set MPI command
        mpi_command = '/usr/lib64/openmpi/bin/mpirun -np'

        command_list = [
            mpi_command,
            str(self.num_procs),
            self.executable,
            self.input_file,
            self.output_file,
        ]

        return ' '.join(filter(None, command_list))

    def assemble_docker_run_command_string(self):
        """  Assemble command list for BACI runin Docker container

            Returns:
                list: command list to execute BACI in Docker container

        """
        command_list = [
            self.sudo,
            " docker run -i --rm -v ",
            self.experiment_dir,
            ":",
            self.experiment_dir,
            " ",
            self.image_name,
            " ",
            self.baci_run_command_string,
        ]

        return ''.join(filter(None, command_list))
