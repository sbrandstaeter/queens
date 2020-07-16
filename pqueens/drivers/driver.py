import sys
import abc
import subprocess
import time
import os
from pqueens.utils.injector import inject
from pqueens.database.mongodb import MongoDB


class Driver(metaclass=abc.ABCMeta):
    """
    Base class for Drivers

    This Driver class is the base class for drivers that actually execute a job on
    a computing resource. It is supposed to unify the interface of drivers and
    fully integrate them in QUEENS to enable testing.

    Attributes:
        simulation_input_template (str): Path to the simulation input file template where
                                         parameters will be parsed
        database (obj): data base object
        output_file (str): Path / name for output files of postprocessor that contain the QoI
                           for the QUEENS analysis
        file_prefix (str): Unique string sequence that is part of the simulation output
                           name and identifies the unprocessed simulation output (that might
                           need to be post-processed).
        output_scratch (str): Path to output base directory that contains the simulation output and
                              postprocessed files
        job (dict): Dictionary containing description of current job to be simulated
        job_id (int): Job ID given in database in range [1, n_jobs]
        batch (int): Job batch number (in case several batch jobs were performed)
        executable (str): Path to the executable that should be used in the QUEENS simulation
        result (np.array): Result of the QUEENS analysis
        postprocessor (str): Path to external postprocessor for the (binary) simulation results
        post_options (lst): List containing settings/options for the external postprocessor
        postpostprocessor (obj): Instance of the PostPost class
        pid (int): Unique process ID for subprocess
        port (int): Port that is used for data forwarding on remote systems
        num_procs (int): Number of MPI-ranks (processors) that should be used for the external
                         main simulation
        num_procs_post (int): Number of MPI-ranks (processors) that should be used for
                              external postprocessing of the simulation data
        experiment_name (str): Name of the current QUEENS analysis. This name is used to
                                construct the naming for the associated directories and files.
        experiment_dir (str): Path to the experiment directory
        input_file (str): Path to current input file for the external simulator

    Returns:
        driver_obj (obj): Instance of the driver class

    """

    def __init__(self, base_settings):
        self.simulation_input_template = base_settings['simulation_input_template']
        # TODO MongoDB object should be passed to init not created within
        self.database = MongoDB(database_address=base_settings['address'])
        self.scheduler_type = base_settings['scheduler_type']
        self.cluster_script = base_settings['cluster_script']
        self.output_file = base_settings['output_file']
        self.file_prefix = base_settings['file_prefix']
        self.output_scratch = base_settings['output_scratch']
        self.job = base_settings['job']
        self.job_id = base_settings['job_id']
        self.batch = base_settings['batch']
        self.executable = base_settings['executable']
        self.result = base_settings['result']
        self.postprocessor = base_settings['postprocessor']
        self.post_options = base_settings['post_options']
        self.postpostprocessor = base_settings['postpostprocessor']
        self.pid = None
        self.port = base_settings['port']
        self.num_procs = base_settings['num_procs']
        self.num_procs_post = base_settings['num_procs_post']
        self.experiment_name = base_settings['experiment_name']
        self.experiment_dir = base_settings['experiment_dir']
        self.input_file = None
        self.input_dic_1 = None

    @classmethod
    def from_config_create_driver(
        cls, config, job_id, batch, port=None, abs_path=None, workdir=None
    ):
        """
        Create driver from problem description

        Args:
            config (dict):      Dictionary with QUEENS problem description
            job_id (int): Job ID given in database in range [1, n_jobs]
            port (int): Port that is used for data forwarding on remote systems
            abs_path (str): Absolute path of postpost-module on the remote (this is depreciated
                            and will be changed, soon)
            batch (int):  Job batch number (in case several batch jobs were performed)
            workdir (str): Path to the working directory for QUEENS on the remote host

        Returns:
            driver: Driver object

        """
        from pqueens.drivers.ansys_driver_native import AnsysDriverNative
        from pqueens.drivers.baci_driver_bruteforce import BaciDriverBruteforce
        from pqueens.drivers.baci_driver_native import BaciDriverNative
        from pqueens.drivers.navierstokes_native import NavierStokesNative
        from pqueens.drivers.baci_driver_deep import BaciDriverDeep
        from pqueens.drivers.baci_driver_docker import BaciDriverDocker
        from pqueens.drivers.openfoam_driver_docker import OpenFOAMDriverDocker

        if abs_path is None:
            from pqueens.post_post.post_post import PostPost

            # FIXME singularity doesnt load post_post form path but rather uses image module
        else:
            import importlib.util

            spec = importlib.util.spec_from_file_location("post_post", abs_path)
            post_post = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(post_post)
            try:
                from post_post.post_post import PostPost
            except ImportError:
                raise ImportError('Could not import the post_post module!')

        driver_dict = {
            'ansys_native': AnsysDriverNative,
            'baci_bruteforce': BaciDriverBruteforce,
            'baci_native': BaciDriverNative,
            'navierstokes_native': NavierStokesNative,
            'baci_deep': BaciDriverDeep,
            'baci_docker': BaciDriverDocker,
            'baci_docker_task': BaciDriverDocker,
            'openfoam_docker': OpenFOAMDriverDocker,
        }
        driver_version = config['driver']['driver_type']
        driver_class = driver_dict[driver_version]

        # ---------------------------- CREATE BASE SETTINGS ---------------------------
        base_settings = {}  # initialize empty dictionary

        # general settings
        base_settings['experiment_name'] = config['experiment_name']

        # scheduler settings
        first = list(config['resources'])[0]
        scheduler_name = config['resources'][first]['scheduler']
        base_settings['scheduler_type'] = config[scheduler_name]['scheduler_type']
        if 'cluster_script' in config[scheduler_name]:
            base_settings['cluster_script'] = config[scheduler_name]['cluster_script']
        else:
            base_settings['cluster_script'] = None
        base_settings['num_procs'] = config[scheduler_name]['num_procs']
        if 'num_procs_post' in config[scheduler_name]:
            base_settings['num_procs_post'] = config[scheduler_name]['num_procs_post']
        else:
            base_settings['num_procs_post'] = 1

        # driver settings
        driver_options = config['driver']['driver_params']
        base_settings['file_prefix'] = driver_options['post_post']['file_prefix']
        base_settings['experiment_dir'] = driver_options['experiment_dir']
        base_settings['job_id'] = job_id
        base_settings['input_file'] = None
        base_settings['simulation_input_template'] = driver_options.get('input_template')
        base_settings['output_file'] = None
        base_settings['output_scratch'] = None
        base_settings['job'] = None
        base_settings['batch'] = batch
        base_settings['executable'] = driver_options['path_to_executable']
        base_settings['result'] = None
        base_settings['port'] = port
        base_settings['postprocessor'] = driver_options.get('path_to_postprocessor', None)
        if base_settings['postprocessor'] is not None:
            base_settings['post_options'] = driver_options['post_process_options']
        else:
            base_settings['post_options'] = None

        # post-post settings
        # TODO "hiding" a complete object in the base settings dict is unbelieveably ugly
        # and should be fixed ASAP
        base_settings['postpostprocessor'] = PostPost.from_config_create_post_post(config)

        # create specific driver
        driver = driver_class.from_config_create_driver(config, base_settings, workdir)

        return driver

    def main_run(self):
        """
        Actual main method of the driver that initializes and runs the executable

        Returns:
            None

        """
        self.prepare_environment()
        self.init_job()
        self.run_job()
        # we take this out of the main run and call in explicitly in remote_main

    # ------------------------ AUXILIARY HIGH-LEVEL METHODS -----------------------
    def prepare_environment(self):
        """
        Prepare the environment for computing by setting up necessary directories and files.

        Returns:
            None

        """
        self.setup_dirs_and_files()

    def finish_and_clean(self):
        """
        Get quantities of interest from postprocessed files and clean up all temporary files.
        General clean-ups like closing ports and saving data.

        Returns:
            None

        """
        if self.postprocessor:
            self.do_postprocessing()
        self.do_postpostprocessing()
        self.finish_job()

    def run_subprocess(self, command_string):
        """
        Method to run command_string outside of Python

        Args:
            command_string (str): Command that should be run externally

        Returns:
            stdout (str): Output of the external subprocess
            stderr (str): Potential errors of the external process
            process_id (str): Process ID of the external process

        """
        process = subprocess.Popen(
            command_string,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            universal_newlines=True,
        )

        stdout, stderr = process.communicate()
        process_id = process.pid
        return stdout, stderr, process_id

    # ------ Base class methods ------------------------------------------------ #
    def init_job(self):
        """
        Initialize job in database

        Returns:
            None

        """
        # Create database object and load the already initiated job entry
        self.job = self.database.load(self.experiment_name, self.batch, 'jobs', {'id': self.job_id})

        # start settings for job
        start_time = time.time()
        self.job['start time'] = start_time

        # create actual input file or dictionaries with parsed parameters
        if self.input_dic_1 is None:
            inject(self.job['params'], self.simulation_input_template, self.input_file)
        else:
            # set path to input dictionary No. 1 and inject
            self.input_file = os.path.join(self.case_dir, self.input_dic_1)
            inject(self.job['params'], self.input_file, self.input_file)

            # set path to input dictionary No. 2 and inject
            self.input_file = os.path.join(self.case_dir, self.input_dic_2)
            inject(self.job['params'], self.input_file, self.input_file)

    def finish_job(self):
        """
        Change status of job to completed in database

        Returns:
            None

        """

        if self.result is None:
            self.job['result'] = None  # TODO: maybe we should better use a pandas format here
            self.job['status'] = 'failed'
            if (
                self.scheduler_type != 'ecs_task'
                and self.scheduler_type != 'local_pbs'
                and self.scheduler_type != 'local_slurm'
            ):
                self.job['end time'] = time.time()
            self.database.save(
                self.job, self.experiment_name, 'jobs', str(self.batch), {'id': self.job_id}
            )
        else:
            self.job['result'] = self.result
            self.job['status'] = 'complete'
            if (
                self.scheduler_type != 'ecs_task'
                and self.scheduler_type != 'local_pbs'
                and self.scheduler_type != 'local_slurm'
            ):
                self.job['end time'] = time.time()
                computing_time = self.job['end time'] - self.job['start time']
                sys.stdout.write(
                    'Successfully completed job {:d} (No. of proc.: {:d}, '
                    'computing time: {:08.2f} s).\n'.format(
                        self.job_id, self.num_procs, computing_time
                    )
                )
            self.database.save(
                self.job, self.experiment_name, 'jobs', str(self.batch), {'id': self.job_id}
            )

    def do_postprocessing(self):
        """
        Trigger an (external) executable that postprocesses (binary) simulation files

        Returns:
            None

        """
        if (
            self.scheduler_type != 'ecs_task'
            and self.scheduler_type != 'local_pbs'
            and self.scheduler_type != 'local_slurm'
        ) or (self.post_options is not None):
            # TODO maybe move to child-class due to specific form (e.g. .dat)
            # TODO the definition of output file and scratch seems redunant as this is already
            # defined in the child class;
            # create input file name
            dest_dir = os.path.join(str(self.experiment_dir), str(self.job_id))
            output_directory = os.path.join(dest_dir, 'output')
            input_file_name = str(self.experiment_name) + '_' + str(self.job_id) + '.dat'
            self.input_file = os.path.join(dest_dir, input_file_name)

            # create output file name
            output_file_name = str(self.experiment_name) + '_' + str(self.job_id)
            self.output_file = os.path.join(output_directory, output_file_name)
            self.output_scratch = self.experiment_name + '_' + str(self.job_id)

            target_file_base_name = os.path.dirname(self.output_file)
            output_file_opt = '--file=' + self.output_file

            if self.post_options:
                for num, option in enumerate(self.post_options):
                    target_file_opt_1 = '--output=' + target_file_base_name
                    target_file_opt_2 = self.file_prefix + "_" + str(num + 1)
                    target_file_opt = os.path.join(target_file_opt_1, target_file_opt_2)
                    postprocessing_list = [
                        self.postprocessor,
                        output_file_opt,
                        option,
                        target_file_opt,
                    ]
                    postprocess_command = ' '.join(filter(None, postprocessing_list))
                    _, _, _ = self.run_subprocess(postprocess_command)
            else:
                target_file_opt = os.path.join(
                    '--output=' + target_file_base_name, self.file_prefix
                )
                postprocessing_list = [self.postprocessor, output_file_opt, target_file_opt]
                postprocess_command = ' '.join(filter(None, postprocessing_list))
                _, _, _ = self.run_subprocess(postprocess_command)

    def do_postpostprocessing(self):
        """
        Extracts necessary information from postprocessed simulation file and saves it to
        the database

        Returns:
            None

        """

        if self.job is None:
            # Load the already initiated job entry
            self.job = self.database.load(
                self.experiment_name, self.batch, 'jobs', {'id': self.job_id}
            )
        dest_dir = os.path.join(str(self.experiment_dir), str(self.job_id))
        output_directory = os.path.join(dest_dir, 'output')
        if self.job['status'] != "failed":
            # this is a security duplicate in case post_post did not catch an error
            self.result = None
            self.result = self.postpostprocessor.postpost_main(output_directory)
            sys.stdout.write("Got result %s\n" % (self.result))

    def setup_mpi(self, ntasks):
        """ Configure and set up the environment for multi_threats """
        pass

    # ---------------- CHILDREN METHODS THAT NEED TO BE IMPLEMENTED ---------------
    @abc.abstractmethod
    def setup_dirs_and_files(self):
        """
        Abstract method for setting up a certain directory and file structure that is necessary
        for deployed driver

        Returns:
            None
        """
        pass

    @abc.abstractmethod
    def run_job(self):
        """
        Abstract method to run the job on computing machine

        Returns:
            None

        """
        pass
