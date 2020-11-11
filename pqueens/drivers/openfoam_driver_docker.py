""" This should be a docstring """

import os
import stat
import shutil
import docker
import getpass
from pqueens.database.mongodb import MongoDB
from pqueens.drivers.driver import Driver
from pqueens.utils.string_extractor_and_checker import extract_string_from_output
from pqueens.utils.injector import inject
from pqueens.utils.run_subprocess import run_subprocess


class OpenFOAMDriverDocker(Driver):
    """ 
    Driver to run OpenFOAM in Docker container

    Attributes:
       docker_image (str): Path to the docker image

    """

    def __init__(self, base_settings):
        # TODO dunder init should not be called with dict
        self.docker_version = base_settings['docker_version']
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
        database_address = 'localhost:27017'
        database_config = dict(
            global_settings=config["global_settings"],
            database=dict(address=database_address, drop_existing=False),
        )
        db = MongoDB.from_config_create_database(database_config)
        base_settings['database'] = db

        base_settings['docker_version'] = config['driver']['driver_type']
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
            using run_subprocess method from utils
        """
        # decide whether task-based run or direct run
        if self.docker_version == 'openfoam_docker_task':
            run_command_string = self.assemble_aws_run_task_command_string()
        else:
            # first alternative (used currently):
            # explicitly assemble run command for Docker container
            run_command_string = self.assemble_docker_run_command_string()

            # second alternative (not used currently): use Docker SDK
            # get Docker client
            # client = docker.from_env()

            # assemble environment for Docker container
            # env = {"USER": getpass.getuser()}

            # assemble volume map for docker container
            # volume_map = {self.experiment_dir: {'bind': self.experiment_dir, 'mode': 'rw'}}

            # run OpenFOAM in Docker container
            # stderr = client.containers.run(self.image_name,
            #                                self.case_run_script,
            #                                remove=True,
            #                                user=os.geteuid(),
            #                                environment=env,
            #                                volumes=volume_map,
            #                                stdout=False,
            #                                stderr=True)

        # run OpenFOAM in Docker container via subprocess
        returncode, self.pid, stdout, stderr = run_subprocess(run_command_string)

        # save AWS ARN and number of processes to database for task-based run
        if self.docker_version == 'openfoam_docker_task':
            self.job['aws_arn'] = extract_string_from_output("taskArn", stdout)
            self.job['num_procs'] = self.num_procs
            self.database.save(
                self.job,
                self.experiment_name,
                'jobs',
                str(self.batch),
                {
                    'id': self.job_id,
                    'expt_dir': self.experiment_dir,
                    'expt_name': self.experiment_name,
                },
            )

        # detection of failed jobs
        if returncode:
            self.result = None
            self.job['status'] = 'failed'

    def assemble_aws_run_task_command_string(self):
        """  Assemble command list for OpenFOAM runin Docker container

            Returns:
                list: command list to execute OpenFOAM in Docker container

        """
        command_list = [
            "aws ecs run-task ",
            "--cluster worker-queens-cluster ",
            "--task-definition docker-queens ",
            "--count 1 ",
            "--overrides '{ \"containerOverrides\": [ {\"name\": \"docker-queens-container\", ",
            "\"command\": [\"",
            self.case_run_script,
            "\"] } ] }'",
        ]

        return ''.join(filter(None, command_list))

    def assemble_docker_run_command_string(self):
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
            "' -v ",
            self.experiment_dir,
            ":",
            self.experiment_dir,
            " ",
            self.image_name,
            " ",
            self.case_run_script,
        ]

        return ''.join(filter(None, command_list))
