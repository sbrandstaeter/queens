import os
import docker
import getpass

from pqueens.database.mongodb import MongoDB
from pqueens.drivers.driver import Driver
from pqueens.utils.string_extractor_and_checker import extract_string_from_output
from pqueens.utils.run_subprocess import run_subprocess


class BaciDriverDocker(Driver):
    """
    Driver to run BACI in Docker container

    Attributes:
        docker_image (str): Path to the docker image

    """

    def __init__(self, base_settings):
        # TODO dunder init should not be called with dict
        self.docker_version = base_settings['docker_version']
        self.docker_image = base_settings['docker_image']
        super(BaciDriverDocker, self).__init__(base_settings)

    @classmethod
    def from_config_create_driver(cls, config, base_settings, workdir=None):
        """ Create Driver from input file

        Args:
            config (dict):          Input options
            base_settings (dict):   Second dict with input options TODO should probably be removed

        Returns:
            driver (obj): BaciDriverDocker object

        """
        database_address = 'localhost:27017'
        database_config = dict(
            global_settings=config["global_settings"],
            database=dict(address=database_address, drop_existing=False),
        )
        db = MongoDB.from_config_create_database(database_config)
        base_settings['database'] = db

        base_settings['docker_version'] = config['driver']['driver_type']
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

        # set destination directory and output prefix
        dest_dir = os.path.join(str(self.experiment_dir), str(self.job_id))
        self.output_prefix = str(self.experiment_name) + '_' + str(self.job_id)

        # generate path to input file
        input_file_str = self.output_prefix + '.dat'
        self.input_file = os.path.join(dest_dir, input_file_str)

        # make output directory if not already existent
        self.output_directory = os.path.join(dest_dir, 'output')
        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory)

        # generate path to output files in general and log file
        self.output_file = os.path.join(self.output_directory, self.output_prefix)
        log_file_str = self.output_prefix + '.out'
        self.log_file = os.path.join(self.output_directory, log_file_str)
        err_file_str = self.output_prefix + '.err'
        self.err_file = os.path.join(self.output_directory, err_file_str)

    def run_job(self):
        """ Actual method to run the job on computing machine
            using run_subprocess method from utils
        """
        # assemble BACI run command string
        baci_run_cmd = self.assemble_baci_run_cmd()

        # decide whether task-based or direct Docker run
        if self.docker_version == 'baci_docker_task':
            run_command_string = self.assemble_aws_task_baci_run_cmd(baci_run_cmd)
        else:
            # first alternative (used currently):
            # explicitly assemble run command for Docker container
            docker_baci_run_cmd = self.assemble_docker_baci_run_cmd(baci_run_cmd)

            # second alternative (not used currently): use Docker SDK
            # get Docker client
            # client = docker.from_env()

            # assemble environment for Docker container
            # env = {"USER": getpass.getuser()}

            # assemble volume map for docker container
            # volume_map = {self.experiment_dir: {'bind': self.experiment_dir, 'mode': 'rw'}}

            # run BACI in Docker container via SDK
            # stderr = client.containers.run(self.image_name,
            #                                self.baci_run_command_string,
            #                                remove=True,
            #                                user=os.geteuid(),
            #                                environment=env,
            #                                volumes=volume_map,
            #                                stdout=False,
            #                                stderr=True)

            if self.scheduler_type == 'nohup':
                # assemble command string for nohup BACI run in Docker container
                run_command_string = self.assemble_nohup_docker_baci_run_cmd(docker_baci_run_cmd)
            else:
                run_command_string = docker_baci_run_cmd

        # run BACI in Docker container via subprocess
        returncode, self.pid, stdout, stderr = run_subprocess(run_command_string)

        # save AWS ARN and number of processes to database for task-based run
        if self.docker_version == 'baci_docker_task':
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
        elif self.scheduler_type == 'nohup':
            # save path to log file and number of processes to database
            self.job['log_file_path'] = self.log_file
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

    def assemble_baci_run_cmd(self):
        """  Assemble command for BACI run

            Returns:
                BACI run command

        """
        # set MPI command
        mpi_cmd = '/usr/lib64/openmpi/bin/mpirun --allow-run-as-root -np'

        command_list = [
            mpi_cmd,
            str(self.num_procs),
            self.executable,
            self.input_file,
            self.output_file,
        ]

        return ' '.join(filter(None, command_list))

    def assemble_aws_task_baci_run_cmd(self, baci_run_cmd):
        """  Assemble command for BACI run as AWS task

            Returns:
                AWS task BACI run command

        """
        command_list = [
            "aws ecs run-task ",
            "--cluster worker-queens-cluster ",
            "--task-definition docker-queens ",
            "--count 1 ",
            "--overrides '{ \"containerOverrides\": [ {\"name\": \"docker-queens-container\", ",
            "\"command\": [\"",
            baci_run_cmd,
            "\"] } ] }'",
        ]

        return ''.join(filter(None, command_list))

    def assemble_docker_baci_run_cmd(self, baci_run_cmd):
        """  Assemble command for BACI run in Docker container

            Returns:
                Docker BACI run command

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
            baci_run_cmd,
        ]

        return ''.join(filter(None, command_list))

    def assemble_nohup_docker_baci_run_cmd(self, docker_baci_run_cmd):
        """  Assemble command for nohup BACI run in Docker container

            Returns:
                nohup Docker BACI run command

        """
        command_list = [
            "nohup",
            docker_baci_run_cmd,
            ">",
            self.log_file,
            "2>",
            self.err_file,
            "< /dev/null &",
        ]

        return ' '.join(filter(None, command_list))
