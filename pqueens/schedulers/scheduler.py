try:
    import simplejson as json
except ImportError:
    import json
import abc
import os
import os.path
import random
import subprocess
import sys
import time
from collections import OrderedDict
from pprint import pprint

import pqueens.interfaces.job_interface as job_interface
from pqueens.drivers.driver import Driver
from pqueens.utils.aws_output_string_extractor import aws_extract
from pqueens.utils.manage_singularity import SingularityManager, hash_files
from pqueens.utils.injector import inject
from pqueens.utils.run_subprocess import run_subprocess
from pqueens.utils.script_generator import generate_submission_script
from pqueens.utils.user_input import request_user_input_with_default_and_timeout


class Scheduler(metaclass=abc.ABCMeta):
    """
    Abstract base class for Schedulers in QUEENS. The purpose of a scheduler
    is twofold: First, it organized the simulations in QUEENS on local or remote computing resources
    by communication with the OS or the scheduling software installed on the system.
    Second it establishes the necessary connections via ssh port-forwarding to enable
    passwordless data-transfer between the database, the localhost and potential remote resources.
    The communication includes also the management of temporary file copies and the build of a
    singularity image, called "driver.simg" that enables the usage of QUEENS on remote clusters.

    Args:
        base_settings (dict): A dictionary containing settings for the base class that are
                              set in the child classes, which are normally created from a
                              config file

    Attributes:
        remote_flag (bool): internal flag that is set to determine whether a setup for remote
                            computing is necessary or not
        config (dict): dictionary that contains the configuration specified in the json input-file
                       for QUEENS
        path_to_singularity (str): (only for remote computing) path to the singularity image on the
                                   remote computing resource
        connect_to_resource (str): (only for remote computing) adress of the remote computing
                                   resource
        port (int): (only for remote computing) port number on the remote which is used for
                    ssh port-forwarding to the database
        job_id (int): internal job ID in QUEENS (used for database organization)
        experiment_name (str): name of the QUEENS simulation
        experiment_dir (str): path to the QUEENS experiment directory
        scheduler_start (str): scheduler specific run command
        cluster_bind (str): directories on the computing resource that should be binded to the
                            singularity image to run the simulation executable from the
                            singularity image
        submission_script_template (str): (only for remote computing) jobscript that should be used
                                          on the remote resource to set up scratch directories
                                          and specific infrastructure for parallel computing
        submission_script_path (str): (only for remote computing) path to which the jobscript should
                                      be copied to on the remote
        scheduler_options (dict): (only for remote computing) further options on flags to configure
                                  the scheduling software on the remote
        no_singularity (bool): (only necessary for local computing) flag to determine whether the
                               local simulations should use singularity (recommended) or are
                               scheduled in sequentially without singularity

    Returns:
        Scheduler (obj): Instance of Scheduler Class

    """

    def __init__(self, base_settings):
        self.remote_flag = base_settings['remote_flag']
        self.config = base_settings['config']
        self.path_to_singularity = base_settings['singularity_path']
        self.connect_to_resource = base_settings['connect']
        self.port = None
        self.job_id = None
        # base settings form child
        self.experiment_name = base_settings['experiment_name']
        self.experiment_dir = base_settings['experiment_dir']
        self.scheduler_start = base_settings['scheduler_start']
        self.cluster_bind = base_settings['cluster_bind']
        self.submission_script_template = base_settings['scheduler_template']
        self.submission_script_path = None  # will be assigned on runtime
        self.scheduler_options = base_settings['scheduler_options']
        self.no_singularity = base_settings['no_singularity']
        self.restart_from_finished_simulation = base_settings['restart_from_finished_simulation']
        self.polling_time = base_settings['polling_time']
        self.docker_image = base_settings['docker_image']
        self.singularity_manager = SingularityManager(
            remote_flag=self.remote_flag,
            connect_to_resource=self.connect_to_resource,
            cluster_bind=self.cluster_bind,
            path_to_singularity=self.path_to_singularity,
        )

    @classmethod
    def from_config_create_scheduler(cls, config, scheduler_name=None):
        """ Create scheduler from problem description

        Args:
            scheduler_name (str): Name of scheduler
            config (dict): Dictionary with QUEENS problem description

        Returns:
            scheduler: scheduler object

        """
        # import here to avoid issues with circular inclusion
        from .local_scheduler import LocalScheduler
        from .PBS_scheduler import PBSScheduler
        from .slurm_scheduler import SlurmScheduler
        from .ecs_task_scheduler import ECSTaskScheduler

        scheduler_dict = {
            'local': LocalScheduler,
            'pbs': PBSScheduler,
            'local_pbs': PBSScheduler,
            'slurm': SlurmScheduler,
            'local_slurm': SlurmScheduler,
            'ecs_task': ECSTaskScheduler,
        }

        if scheduler_name is not None:
            scheduler_options = config[scheduler_name]
        else:
            scheduler_options = config['scheduler']

        # determine which object to create
        scheduler_class = scheduler_dict[scheduler_options["scheduler_type"]]

        # ---------------------------- CREATE BASE SETTINGS ---------------------------
        base_settings = {}  # initialize empty dictionary

        # general settings
        base_settings['experiment_name'] = config['global_settings']['experiment_name']
        base_settings['polling_time'] = config.get('polling-time', 1)

        # scheduler settings
        base_settings['options'] = scheduler_options
        if (
            scheduler_options["scheduler_type"] == 'local'
            or scheduler_options["scheduler_type"] == 'local_pbs'
            or scheduler_options["scheduler_type"] == 'local_slurm'
            or scheduler_options["scheduler_type"] == 'ecs_task'
        ):
            base_settings['remote_flag'] = False
            base_settings['singularity_path'] = None
            base_settings['connect'] = None
            base_settings['cluster_bind'] = config['driver']['driver_params'].get('cluster_bind')
            base_settings['scheduler_template'] = None
        elif (
            scheduler_options["scheduler_type"] == 'pbs'
            or scheduler_options["scheduler_type"] == 'slurm'
        ):
            base_settings['remote_flag'] = True
            base_settings['singularity_path'] = config['driver']['driver_params'][
                'path_to_singularity'
            ]
            base_settings['connect'] = config[scheduler_name]['connect_to_resource']
            base_settings['cluster_bind'] = config['driver']['driver_params']['cluster_bind']
        else:
            raise RuntimeError(
                "Scheduler type was not specified correctly! "
                "Choose either 'local', 'pbs', 'slurm' or 'ecs_task'!"
            )

        # driver settings
        driver_options = config['driver']['driver_params']
        base_settings['experiment_dir'] = driver_options['experiment_dir']
        # get docker image if present
        if driver_options.get('docker_image', False):
            base_settings['docker_image'] = driver_options['docker_image']
        else:
            base_settings['docker_image'] = None
        # make no singularity the default
        if driver_options.get('no_singularity', True):
            base_settings['no_singularity'] = True
        else:
            base_settings['no_singularity'] = False
        base_settings['restart_from_finished_simulation'] = driver_options.get(
            'restart_from_finished_simulation', False
        )

        # config settings
        base_settings['config'] = config
        # ----------------------------- END BASE SETTINGS -----------------------------

        # create specific scheduler
        scheduler = scheduler_class.from_config_create_scheduler(
            config, base_settings, scheduler_name=None
        )
        return scheduler

    def pre_run(self):
        """
        A wrapper method that sets-up the scheduling configuration for local
        and remote computing, such as automated port-forwarding, necessary addresses
        or singularity settings

        Returns:
            None

        """
        if self.remote_flag:
            _, _, hostname, _ = run_subprocess('hostname -i')
            _, _, username, _ = run_subprocess('whoami')
            address_localhost = username.rstrip() + r'@' + hostname.rstrip()

            self.kill_previous_queens_ssh_remote(username)
            self.establish_port_forwarding_local(address_localhost)
            self.establish_port_forwarding_remote(address_localhost)
            self.copy_temp_json()
            self.copy_post_post()

            self.singularity_manager.check_singularity_system_vars()
            self.singularity_manager.prepare_singularity_files()
        else:
            if not self.no_singularity:
                self.singularity_manager.check_singularity_system_vars()
                self.singularity_manager.prepare_singularity_files()

    def post_run(self):  # will actually be called in job_interface
        """
        A finishing method for the scheduler module that takes care of closing
        utilized ports and connections.

        Returns:
            None

        """
        if self.remote_flag:
            self.close_remote_port()
            print('All port-forwardings were closed again.')
        else:
            pass

    # ------------------------ AUXILIARY HIGH LEVEL METHODS -----------------------
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
            self.connect_to_resource,
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
                        command_list = ['ssh', self.connect_to_resource, '\'kill -9', ssh_id + '\'']
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
            self.port = random.randrange(2030, 20000, 1)
            command_list = [
                'ssh',
                '-t',
                '-t',
                self.connect_to_resource,
                '\'ssh',
                '-fN',
                '-g',
                '-L',
                str(self.port) + r':' + 'localhost' + r':27017',
                address_localhost + '\'',
            ]
            command_string = ' '.join(command_list)
            port_fail = os.popen(command_string).read()
            time.sleep(0.1)
        print('Remote port-forwarding successfully established for port %s' % (self.port))

    def establish_port_forwarding_local(self, address_localhost):
        """
        Establish a port-forwarding for localhost's port 9001 to the remote's ssh-port 22
        for passwordless communication with the remote machine over ssh

        Args:
            address_localhost (str): IP-address of the localhost

        Returns:
            None

        """
        remote_address = self.connect_to_resource.split(r'@')[1]
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

    def close_remote_port(self):
        """
        Closes the ports that were used in the current QUEENS simulation for remote computing.

        Returns:
            None

        """
        # get the process id of open port
        _, _, username, _ = run_subprocess('whoami')
        command_list = [
            'ssh',
            self.connect_to_resource,
            '\'ps -aux | grep ssh | grep',
            username.rstrip(),
            '| grep',
            str(self.port) + ':localhost:27017\'',
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
                command_list = ['ssh', self.connect_to_resource, '\'kill -9', ssh_id + '\'']
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
            self.config['input_file'],
            self.connect_to_resource + ':' + self.path_to_singularity + '/temp.json',
        ]
        command_string = ' '.join(command_list)
        _, _, _, stderr = run_subprocess(command_string)
        if stderr:
            raise RuntimeError("Error! Was not able to copy post_post class to remote! Abort...")

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
            self.connect_to_resource,
            '\'rm -r',
            self.path_to_singularity + '/post_post\'',
        ]
        command_string = ' '.join(command_list)
        _, _, _, _ = run_subprocess(command_string)

        # copy new files
        command_list = [
            "scp -r",
            abs_path_post_post,
            self.connect_to_resource + ':' + self.path_to_singularity + '/post_post',
        ]
        command_string = ' '.join(command_list)
        _, _, _, stderr = run_subprocess(command_string)
        if stderr:
            raise RuntimeError(
                "Error! Was not able to copy post_post directory to remote! Abort..."
            )

    def submit(self, job_id, batch):
        """ Function to submit new job to scheduling software on a given resource

        Args:
            job_id (int):            ID of job to submit
            batch (int):             Batch number of job

        Returns:
            pid (int):               process id of job

        """

        submitter = self.get_submitter()
        pid = submitter(job_id, batch)

        return pid

    def submit_post_post(self, job_id, batch):
        """ Function to submit new post-post job to scheduling software on a given resource

        Args:
            job_id (int):            ID of job to submit
            batch (int):             Batch number of job

       """
        # create driver
        driver_obj = Driver.from_config_create_driver(self.config, job_id, batch)

        # do post-processing (if required), post-post-processing,
        # finish and clean job
        driver_obj.finish_and_clean()

    def get_submitter(self):
        """Get function for submission of job.

        Which function should be used depends on whether or not restart is performed, on the
        computing resource and whether or not singularity is used.

        Returns:
            function object:         function for submission of job

        """
        if job_interface.restart_flag:
            if self.remote_flag:
                return self._restart_remote
            else:
                if self.no_singularity:
                    return self._restart_local
                else:
                    return self._restart_local_singularity

        else:
            if self.remote_flag:
                return self._submit_remote
            else:
                if self.no_singularity:
                    return self._submit_local
                else:
                    return self._submit_local_singularity

    # ------- CHILDREN METHODS THAT NEED TO BE IMPLEMENTED / ABSTRACTMETHODS ------
    @abc.abstractmethod  # how to check this is dependent on cluster / env
    def alive(self, process_id):
        pass

    @abc.abstractmethod
    def get_cluster_job_id(self, output):
        pass

    # ------------- private helper methods ----------------#
    def _submit_remote(self, job_id, batch):
        """Submit job on remote with singularity.

        Args:
            job_id (int):    ID of job to submit
            batch (str):     Batch number of job

        Returns:
            int:            process ID

        """
        # set data for jobscsript
        self.scheduler_options[
            'INPUT'
        ] = '--job_id={} --batch={} --port={} --path_json={} --workdir '.format(
            job_id, batch, self.port, self.path_to_singularity
        )
        self.scheduler_options['EXE'] = self.path_to_singularity + '/driver.simg'
        self.scheduler_options['job_name'] = '{}_{}_{}'.format(
            self.experiment_name, 'queens', job_id
        )
        dest_dir = str(self.experiment_dir) + '/' + str(job_id) + "/output"
        self.scheduler_options['DESTDIR'] = dest_dir
        self.scheduler_options['OUTPUTPREFIX'] = ''
        self.scheduler_options['POSTPOSTPROCESSFLAG'] = 'true'
        # following options only required for non-Singularity-based run
        self.scheduler_options['POSTPROCESSFLAG'] = 'false'
        self.scheduler_options['POSTEXE'] = ''
        self.scheduler_options['nposttasks'] = ''

        # generate job script for submission
        self.submission_script_path = os.path.join(self.experiment_dir, 'jobfile.sh')
        generate_submission_script(
            self.scheduler_options,
            self.submission_script_path,
            self.submission_script_template,
            self.connect_to_resource,
        )

        # submit the job with job_script.sh
        cmdlist_remote_main = [
            'ssh',
            self.connect_to_resource,
            '"cd',
            self.experiment_dir,
            ';',
            self.scheduler_start,
            self.submission_script_path,
            '"',
        ]
        cmd_remote_main = ' '.join(cmdlist_remote_main)
        _, _, stdout, stderr = run_subprocess(cmd_remote_main)
        if stderr:
            raise RuntimeError(
                "\nThe file 'remote_main' in remote singularity image "
                "could not be executed properly!"
                f"\nStderr from remote:\n{stderr}"
            )
        match = self.get_cluster_job_id(stdout)

        try:
            return int(match)
        except ValueError:
            sys.stderr.write(stdout)
            return None

    def _submit_local(self, job_id, batch):
        """Submit job locally, either natively or with job script or with task script.

        Args:
            job_id (int):    ID of job to submit
            batch (str):     Batch number of job

        Returns:
            driver_obj.pid (int): process ID
        """

        # only required for ECS task scheduler:
        # check whether new task definition is required
        if type(self).__name__ == 'ECSTaskScheduler':
            # check image and container path in most recent revision of task
            # definition 'docker-qstdoutueens', if existent
            new_task_definition_required = False
            cmd = 'aws ecs describe-task-definition --task-definition docker-queens'
            _, _, stdout, stderr = run_subprocess(cmd)

            # new task definition required, since definition 'docker-queens' is not existent
            if stderr:
                new_task_definition_required = True

            # check image and container path in most recent revision of task
            # definition 'docker-queens'
            image_str = aws_extract("image", stdout)
            container_path_str = aws_extract("containerPath", stdout)

            # new task definition required, since definition 'docker-queens'
            # is not equal to desired one
            if image_str != str(self.docker_image) or container_path_str != str(
                self.experiment_dir
            ):
                new_task_definition_required = True

            # submit new task definition
            if new_task_definition_required:
                # set data for task definition
                self.scheduler_options['EXPDIR'] = str(self.experiment_dir)
                self.scheduler_options['IMAGE'] = str(self.docker_image)

                # Parse data to submission script template
                self.submission_script_path = os.path.join(self.experiment_dir, 'taskscript.json')
                generate_submission_script(
                    self.scheduler_options,
                    self.submission_script_path,
                    self.submission_script_template,
                )

                # change directory
                os.chdir(self.experiment_dir)

                # register task definition
                cmdlist = [
                    "aws ecs register-task-definition --cli-input-json file://",
                    self.submission_script_path,
                ]
                cmd = ''.join(cmdlist)

                _, _, _, stderr = run_subprocess(cmd)
                if stderr:
                    raise RuntimeError(
                        "\nThe task definition could not be registered properly!"
                        f"\nStderr:\n{stderr}"
                    )

        # create and run driver
        driver_obj = Driver.from_config_create_driver(self.config, job_id, batch)
        driver_obj.main_run()

        # only required for local scheduler: finish-and-clean call
        # (taken care of by submit_post_post for other schedulers)
        if type(self).__name__ == 'LocalScheduler':
            driver_obj.finish_and_clean()

        return driver_obj.pid

    def _submit_local_singularity(self, job_id, batch):
        """Submit job locally with singularity. And allow parallel execution.

        Args:
            job_id (int): ID of job to submit
            batch (str): Batch number of job

        Returns:
            pid (int): process ID

        """
        local_path_json = self.config['input_file']
        remote_args = '--job_id={} --batch={} --port={} --path_json={}'.format(
            job_id, batch, '000', local_path_json
        )
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        rel_path = '../../driver.simg'
        local_singularity_path = os.path.join(script_dir, rel_path)
        cmdlist_remote_main = ['/usr/bin/singularity run', local_singularity_path, remote_args]
        cmd_remote_main = ' '.join(cmdlist_remote_main)

        _, pid, _, _ = run_subprocess(cmd_remote_main, subprocess_type='submit')

        return pid

    def _restart_remote(self, job_id, batch):
        """Restart job on remote with singularity.

        Args:
            job_id (int):    ID of job to submit
            batch (str):     Batch number of job

        Returns:
            int:            process ID is 0 for restart
        """
        self.scheduler_options['EXE'] = self.path_to_singularity + '/driver.simg'
        self.scheduler_options['INPUT'] = '--job_id={} --batch={} --port={} --path_json={}'.format(
            job_id, batch, self.port, self.path_to_singularity
        )
        command_list = [
            'singularity run',
            self.scheduler_options['EXE'],
            self.scheduler_options['INPUT'],
            '--post=true',
        ]
        self.submission_script_path = ' '.join(command_list)
        cmdlist_remote_main = [
            'ssh',
            self.connect_to_resource,
            '"cd',
            self.experiment_dir,
            ';',
            self.submission_script_path,
            '"',
        ]
        cmd_remote_main = ' '.join(cmdlist_remote_main)
        _, _, _, _ = run_subprocess(cmd_remote_main)

        return 0

    def _restart_local(self, job_id, batch):
        """Restart job locally without singularity.

        Args:
            job_id (int):    ID of job to submit
            batch (str):     Batch number of job

        Returns:
            int:            process ID is 0 for restart
        """
        driver_obj = Driver.from_config_create_driver(self.config, job_id, batch)

        driver_obj.finish_and_clean()

        return 0

    def _restart_local_singularity(self, job_id, batch):
        """Restart job locally with singularity.

        Args:
            job_id (int):    ID of job to submit
            batch (str):     Batch number of job

        Returns:
            int:            process ID is 0 for restart
        """
        local_path_json = self.config['input_file']
        remote_args = '--job_id={} --batch={} --port={} --path_json={}'.format(
            job_id, batch, '000', local_path_json
        )
        script_dir = os.path.dirname(__file__)
        rel_path = '../../driver.simg'
        local_singularity_path = os.path.join(script_dir, rel_path)
        cmdlist_remote_main = [
            '/usr/bin/singularity run',
            local_singularity_path,
            remote_args,
            '--post=true',
        ]
        cmd_remote_main = ' '.join(cmdlist_remote_main)
        _, _, _, _ = run_subprocess(cmd_remote_main)

        return 0
