""" This should be a module docstring """

try:
    import simplejson as json
except ImportError:
    import json
import abc
from collections import OrderedDict
import os
import os.path
import hashlib
import subprocess
import sys
from pqueens.utils.injector import inject
from pqueens.drivers.driver import Driver


class Scheduler(metaclass=abc.ABCMeta):
    """ Base class for schedulers """

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

    @classmethod
    def from_config_create_scheduler(cls, config, scheduler_name=None):
        """ Create scheduler from problem description

        Args:
            scheduler_name (string): Name of scheduler
            config (dict): Dictionary with QUEENS problem description

        Returns:
            scheduler: scheduler object

        """
        # import here to avoid issues with circular inclusion
        from .local_scheduler import LocalScheduler
        from .PBS_scheduler import PBSScheduler
        from .slurm_scheduler import SlurmScheduler

        scheduler_dict = {'local': LocalScheduler, 'pbs': PBSScheduler, 'slurm': SlurmScheduler}

        if scheduler_name is not None:
            scheduler_options = config[scheduler_name]
        else:
            scheduler_options = config['scheduler']

        # determine which object to create
        scheduler_class = scheduler_dict[scheduler_options["scheduler_type"]]

        # ---------------------------- CREATE BASE SETTINGS ---------------------------
        base_settings = {}  # initialize empty dictionary
        base_settings['options'] = scheduler_options
        base_settings['experiment_name'] = config['global_settings']['experiment_name']
        driver_options = config['driver']['driver_params']
        base_settings['experiment_dir'] = driver_options['experiment_dir']
        if config['driver']['driver_params'].get('no_singularity'):
            base_settings['no_singularity'] = True
        else:
            base_settings['no_singularity'] = False

        if scheduler_options["scheduler_type"] == 'local':
            base_settings['remote_flag'] = False
            base_settings['singularity_path'] = None
            base_settings['connect'] = None
            base_settings['scheduler_template'] = None
            base_settings['cluster_bind'] = config['driver']['driver_params']['cluster_bind']
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
                "Slurm type was not specified correctly! Choose either 'local', 'pbs' or 'slurm'!"
            )

        base_settings['config'] = config
        # ----------------------------- END BASE SETTINGS -----------------------------

        scheduler = scheduler_class.from_config_create_scheduler(
            config, base_settings, scheduler_name=None
        )
        return scheduler

    def pre_run(self):
        """ This should be a docstring """
        if self.remote_flag:
            hostname, _, _ = self.run_subprocess('hostname -i')
            username, _, _ = self.run_subprocess('whoami')
            address_localhost = username.rstrip() + r'@' + hostname.rstrip()

            # self.kill_user_ssh_remote(username) TODO: think about this option
            self.establish_port_forwarding_local(address_localhost)
            self.establish_port_forwarding_remote(address_localhost)
            self.copy_temp_json()
            self.copy_post_post()

        if not self.no_singularity:
            self.check_singularity_system_vars()
            self.prepare_singularity_files()

    def post_run(self):  # will actually be called in job_interface
        """ This should be a docstring """
        if self.remote_flag:
            self.close_remote_port()
        else:
            pass

    # ------------------------ AUXILIARY HIGH LEVEL METHODS -----------------------
    def kill_user_ssh_remote(self, username):
        """ Docstring """
        command_list = ['ssh', self.connect_to_resource, 'killall -u', username, 'ssh\'']
        command_string = ' '.join(command_list)
        _, _, _ = self.run_subprocess(command_string)

    def check_singularity_system_vars(self):
        """Docstring"""
        # Check if SINGULARITY_BIND exists and if not write it to .bashrc file
        if self.remote_flag:
            command_list = ['ssh', self.connect_to_resource, '\'echo $SINGULARITY_BIND\'']
        else:
            command_list = ['echo $SINGULARITY_BIND']
        command_string = ' '.join(command_list)
        stdout, stderr, _ = self.run_subprocess(command_string)
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
        stdout, stderr, _ = self.run_subprocess(command_string)

        # Create a Singularity PATH variable that is equal to the host PATH
        if self.remote_flag:
            command_list = ['ssh', self.connect_to_resource, '\'echo $SINGULARITYENV_APPEND_PATH\'']
        else:
            command_list = ['echo $SINGULARITYENV_APPEND_PATH']
        command_string = ' '.join(command_list)
        stdout, stderr, _ = self.run_subprocess(command_string)
        if stdout == "\n":
            if self.remote_flag:
                command_list = [
                    'ssh',
                    self.connect_to_resource,
                    "\"echo 'export SINGULARITYENV_APPEND_PATH=$PATH' >> ~/.bashrc && source ~/.bashrc\"",
                ]  # noqa
            else:
                command_list = [
                    "echo 'export SINGULARITYENV_APPEND_PATH=$PATH' >> ~/.bashrc && source ~/.bashrc"
                ]  # noqa
            command_string = ' '.join(command_list)
            stdout, stderr, _ = self.run_subprocess(command_string)

        # Create a Singulartity LD_LIBRARY_PATH variable that is equal to the host LD_LIBRARY_PATH
        if self.remote_flag:
            command_list = [
                'ssh',
                self.connect_to_resource,
                '\'echo $SINGULARITYENV_APPEND_LD_LIBRARY_PATH\'',
            ]
        else:
            command_list = ['echo $SINGULARITYENV_APPEND_LD_LIBRARY_PATH']
        command_string = ' '.join(command_list)
        stdout, stderr, _ = self.run_subprocess(command_string)
        if stdout == "\n":
            if self.remote_flag:
                command_list = [
                    'ssh',
                    self.connect_to_resource,
                    "\"echo 'export SINGULARITYENV_APPEND_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' >> ~/.bashrc && source ~/.bashrc\"",
                ]  # noqa
            else:
                command_list = [
                    "echo 'export SINGULARITYENV_APPEND_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' >> ~/.bashrc && source ~/.bashrc"
                ]  # noqa
            command_string = ' '.join(command_list)
            stdout, stderr, _ = self.run_subprocess(command_string)

    def establish_port_forwarding_remote(self, address_localhost):
        """ docstring """
        self.port = 2030  # just a start value to check for next open port
        port_fail = 1

        while port_fail != "":
            self.port += 1
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
        print('Remote port-forwarding sucessfully established for port %s' % (self.port))

    def establish_port_forwarding_local(self, address_localhost):
        """ docstring """
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
        # TODO Think of some kind of error catching here; so far it works but error might be cryptical

    def close_remote_port(self):  # TODO: Check if this might interfer with the current framework
        """ docs """
        command_string = (
            'ssh ' + self.connect_to_resource + ' "kill $(lsof -t -i:' + str(self.port) + ')"'
        )
        _, stderr, _ = self.run_subprocess(command_string)

    def copy_temp_json(self):
        """ docstrig """
        command_list = [
            "scp",
            self.config['input_file'],
            self.connect_to_resource + ':' + self.path_to_singularity + '/temp.json',
        ]
        command_string = ' '.join(command_list)
        stdout, stderr, _ = self.run_subprocess(command_string)
        if stderr:
            raise RuntimeError("Error! Was not able to copy post_post class to remote! Abort...")

    def copy_post_post(self):
        """ docstring """
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
        _, _, _ = self.run_subprocess(command_string)

        # copy new files
        command_list = [
            "scp -r",
            abs_path_post_post,
            self.connect_to_resource + ':' + self.path_to_singularity + '/post_post',
        ]
        command_string = ' '.join(command_list)
        stdout, stderr, _ = self.run_subprocess(command_string)
        if stderr:
            raise RuntimeError(
                "Error! Was not able to copy post_post directory to remote! Abort..."
            )

    def create_singularity_image(self):
        """ Add current environment to predesigned singularity container for cluster applications """
        # create hash for files in image
        self.hash_files('hashing')
        # create the actual image
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        rel_path1 = '../../driver.simg'
        rel_path2 = '../../singularity_recipe'
        abs_path1 = os.path.join(script_dir, rel_path1)
        abs_path2 = os.path.join(script_dir, rel_path2)
        command_list = ["sudo /usr/bin/singularity build", abs_path1, abs_path2]
        command_string = ' '.join(command_list)
        _, stderr, _ = self.run_subprocess(command_string)
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        rel_path = '../../driver.simg'
        abs_path = os.path.join(script_dir, rel_path)
        if not os.path.isfile(abs_path):
            raise RuntimeError(
                '''Build of local singularity image failed!
              This could have several reasons but make sure to run QUEENS from the base
              directory containing the main.py file to set the proper relatives paths!'''
            )

    def hash_files(self, mode=None):
        """ docstring """
        hashlist = []
        hasher = hashlib.md5()
        # hash all drivers
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        rel_path = "../drivers"
        abs_path = os.path.join(script_dir, rel_path)
        elements = os.listdir(abs_path)
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
        # write hash list to a file in utils directory
        if mode is not None:
            rel_path = '../../hashfile.txt'
            abs_path = os.path.join(script_dir, rel_path)
            with open(abs_path, 'w') as myfile:
                for item in hashlist:
                    myfile.write("%s" % item)
        else:
            return hashlist

    def prepare_singularity_files(self):
        """ docstring """
        # check existence local
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        rel_path = '../../driver.simg'
        abs_path = os.path.join(script_dir, rel_path)
        if os.path.isfile(abs_path):
            # check singularity status local
            rel_path = '../../hashfile.txt'
            abs_path = os.path.join(script_dir, rel_path)
            with open(abs_path, 'r') as oldhash:
                old_data = oldhash.read()
            hashlist = self.hash_files()
            # Write local singularity image and remote image
            if old_data != ''.join(hashlist):
                print(
                    "Local singularity image is not up-to-date with QUEENS! Writing new local image..."
                )
                print("(This will take 3 min or so, but needs only to be done once)")
                # deleting old image
                rel_path = '../../driver*'
                abs_path = os.path.join(script_dir, rel_path)
                command_list = ['rm', abs_path]
                command_string = ' '.join(command_list)
                _, _, _ = self.run_subprocess(command_string)
                self.create_singularity_image()
                print("Local singularity image written sucessfully!")
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
                    stdout, stderr, _ = self.run_subprocess(command_string)
                    if stderr:
                        raise RuntimeError(
                            "Error! Was not able to copy local singulariy image to remote! Abort..."
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
                stdout, stderr, _ = self.run_subprocess(command_string)
                if 'N' in stdout:
                    # Update remote image
                    print(
                        "Remote singularity image is not existend! Updating remote image from local image..."
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
                    stdout, stderr, _ = self.run_subprocess(command_string)
                    if stderr:
                        raise RuntimeError(
                            "Error! Was not able to copy local singulariy image to remote! Abort..."
                        )
                    print('All singularity images ok! Starting simulation on cluster...')

        else:
            # local image was not even existend --> create local and remote image
            print("No local singularity image found! Building new image...")
            print("(This will take 3 min or so, but needs only to be done once)")
            print("_______________________________________________________________________________")
            print("")
            print("Make sure QUEENS was called from the base directory containing the main.py file")
            print("to set the correct relative paths for the image; otherwise abort!")
            print("_______________________________________________________________________________")
            self.create_singularity_image()
            print("Local singularity image written sucessfully!")
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
                stdout, stderr, _ = self.run_subprocess(command_string)
                if stderr:
                    raise RuntimeError(
                        "Error! Was not able to copy local singulariy image to remote! Abort..."
                    )
                print('All singularity images ok! Starting simulation on cluster...')

    def run_subprocess(self, command_string):
        """ Method to run command_string outside of Python """
        process = subprocess.Popen(
            command_string,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            universal_newlines=True,
        )

        stdout, stderr = process.communicate()
        return stdout, stderr, process

    def submit(self, job_id, batch):
        """ Function to submit new job to scheduling software on a given resource


        Args:
            job_id (int):               Id of job to submit
            experiment_name (string):   Name of experiment
            batch (string):             Batch number of job
            experiment_dir (string):    Directory of experiment
            database_address (string):  Address of database to connect to
            driver_options (dict):      Options for driver

        Returns:
            int: proccess id of job

        """
        if self.remote_flag:
            self.scheduler_options[
                'INPUT'
            ] = '--job_id={} --batch={} --port={} --path_json={}'.format(
                job_id, batch, self.port, self.path_to_singularity
            )
            self.scheduler_options['EXE'] = self.path_to_singularity + '/driver.simg'
            self.scheduler_options['job_name'] = '{}_{}'.format(self.experiment_name, job_id)

            # Parse data to job_scheduler_template
            self.create_submission_script(job_id)

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
            stdout, stderr, _ = self.run_subprocess(cmd_remote_main)
            match = self.get_process_id_from_output(stdout)
            try:
                return int(match)
            except ValueError:
                sys.stderr.write(stdout)
                return None

            if stderr:
                raise RuntimeError(
                    "The file 'remote_main' in remote singularity image could not be executed properly!"
                )
        else:
            if self.no_singularity:
                with open(self.config['input_file'], 'r') as myfile:
                    config = json.load(myfile, object_pairs_hook=OrderedDict)

                driver_obj = Driver.from_config_create_driver(config, job_id, batch)
                # Run the singularity image in just one step
                driver_obj.main_run()
                driver_obj.finish_and_clean()
                return driver_obj.pid
            else:
                local_path_json = self.config['input_file']
                remote_args = '--job_id={} --batch={} --port={} --path_json={}'.format(
                    job_id, batch, '000', local_path_json
                )
                script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
                rel_path = '../../driver.simg'
                local_singularity_path = os.path.join(script_dir, rel_path)
                cmdlist_remote_main = [
                    '/usr/bin/singularity run',
                    local_singularity_path,
                    remote_args,
                ]
                cmd_remote_main = ' '.join(cmdlist_remote_main)
                # stdout, stderr, _ = self.run_subprocess(cmd_remote_main)
                process = subprocess.Popen(
                    cmd_remote_main, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                pid = process.pid
                return pid

    def create_submission_script(self, job_id):
        dest_dir = str(self.experiment_dir) + '/' + str(job_id) + "/output"
        self.scheduler_options['DESTDIR'] = dest_dir
        self.submission_script_path = str(self.experiment_dir) + '/jobfile.sh'

        # local dummy path
        local_dummy_path = os.path.join(os.path.dirname(__file__), 'dummy_jobfile')
        # create actual submission file with parsed parameters
        inject(self.scheduler_options, self.submission_script_template, local_dummy_path)
        # copy submission script to cluster on specified location
        command_list = [
            'scp',
            local_dummy_path,
            self.connect_to_resource + ':' + self.submission_script_path,
        ]
        command_string = ' '.join(command_list)
        stdout, stderr, p = self.run_subprocess(command_string)
        # delete local dummy jobfile
        command_list = ['rm', local_dummy_path]
        command_string = ' '.join(command_list)
        stdout, stderr, p = self.run_subprocess(command_string)

    # ------- CHILDREN METHODS THAT NEED TO BE IMPLEMENTED / ABSTRACTMETHODS ------
    @abc.abstractmethod  # how to check this is dependent on cluster / env
    def alive(self, process_id):
        """ docstring """
        pass

    @abc.abstractmethod
    def get_process_id_from_output(self, output):
        pass

    @abc.abstractmethod  # how to check this is dependent on cluster / env
    def alive(self, process_id):
        pass
