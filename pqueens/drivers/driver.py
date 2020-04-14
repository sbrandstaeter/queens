""" There should be a docstring """

import sys
import abc
import subprocess
import time
import os
from pqueens.utils.injector import inject
from pqueens.database.mongodb import MongoDB


class Driver(metaclass=abc.ABCMeta):

    """ Base class for Drivers

    This Driver class is the base class for drivers that actually execute a job on
    a computing resource. It is supposed to unify the interface of drivers and
    fully integrate them in QUEENS to enable testing. Furthermore, an abstract
    driver class will give rise to the usage of singularity containers for HPC
    applications.

    Attributes:

    """

    def __init__(self, base_settings):
        self.template = base_settings['template']
        # TODO MongoDB object should be passed to init not created within
        self.database = MongoDB(database_address=base_settings['address'])
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
        self.mpi_flags = {}  # config mpi for each machine (build on runtime)
        self.pid = None
        self.port = base_settings['port']
        self.num_procs = base_settings['num_procs']
        self.num_procs_post = base_settings['num_procs_post']
        self.experiment_name = base_settings['experiment_name']
        self.experiment_dir = base_settings['experiment_dir']
        self.input_file = None

    @classmethod
    def from_config_create_driver(
        cls, config, job_id, batch, port=None, abs_path=None, workdir=None
    ):
        """ Create driver from problem description

        Args:
            config (dict):      Dictionary with QUEENS problem description
            driver_type (str):  Name of driver to identify right section in options
                                dict (optional)
            database (database):database to use (optional)

        Returns:
            driver: Driver object

        """
        from pqueens.drivers.ansys_driver_native import AnsysDriverNative
        from pqueens.drivers.baci_driver_bruteforce import BaciDriverBruteforce
        from pqueens.drivers.baci_driver_native import BaciDriverNative
        from pqueens.drivers.navierstokes_native import NavierStokesNative
        from pqueens.drivers.baci_driver_deep import BaciDriverDeep

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
        }
        driver_version = config['driver']['driver_type']
        driver_class = driver_dict[driver_version]

        # ---------------------------- CREATE BASE SETTINGS ---------------------------
        driver_options = config['driver']['driver_params']
        first = list(config['resources'])[0]
        scheduler_name = config['resources'][first]['scheduler']
        base_settings = {}
        base_settings['experiment_name'] = config['experiment_name']
        base_settings['num_procs'] = config[scheduler_name]['num_procs']
        base_settings['file_prefix'] = driver_options['post_post']['file_prefix']
        if 'num_procs_post' in config[scheduler_name]:
            base_settings['num_procs_post'] = config[scheduler_name]['num_procs_post']
        else:
            base_settings['num_procs_post'] = 1
        base_settings['experiment_dir'] = driver_options['experiment_dir']
        base_settings['job_id'] = job_id
        base_settings['input_file'] = None
        base_settings['template'] = driver_options['input_template']
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

        # TODO "hiding" a complete object in the base settings dict is unbelieveably ugly
        # and should be fixed ASAP
        base_settings['postpostprocessor'] = PostPost.from_config_create_post_post(config)
        driver = driver_class.from_config_create_driver(config, base_settings, workdir)

        return driver

    def main_run(self):
        """ Actual main method of the driver """
        self.prepare_environment()
        self.init_job()
        self.run_job()
        # we take this out of the main run and call in explicitly in remote_main
        # self.finish_and_clean()

    # ------------------------ AUXILIARY HIGH-LEVEL METHODS -----------------------
    def prepare_environment(self):
        """ Prepare the environment for computing """
        self.setup_dirs_and_files()

    def finish_and_clean(self):
        """ Finish and clean the resources and environment """
        if self.postprocessor:
            self.do_postprocessing()
        self.do_postpostprocessing()
        self.finish_job()

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
        process_id = process.pid
        return stdout, stderr, process_id

    # ------ Base class methods ------------------------------------------------ #
    def init_job(self):
        """ Initialize job in database

            Returns:
                dict: Dictionary with job information

        """
        # Create database object and load the already initiated job entry
        self.job = self.database.load(self.experiment_name, self.batch, 'jobs', {'id': self.job_id})

        # start settings for job
        start_time = time.time()
        self.job['start time'] = start_time

        # sys.stderr.write("Job launching after %0.2f seconds in submission.\n"
        #                 % (start_time-self.job['submit time']))

        # create actual input file with parsed parameters
        inject(self.job['params'], self.template, self.input_file)

    def finish_job(self):
        """ Change status of job to completed in database """

        if self.result is None:
            self.job['result'] = None  # TODO: maybe we should better use a pandas format here
            self.job['status'] = 'failed'
            self.job['end time'] = time.time()
            self.database.save(
                self.job, self.experiment_name, 'jobs', str(self.batch), {'id': self.job_id}
            )
        else:
            self.job['result'] = self.result
            self.job['status'] = 'complete'
            self.job['end time'] = time.time()
            self.database.save(
                self.job, self.experiment_name, 'jobs', str(self.batch), {'id': self.job_id}
            )

    def do_postprocessing(self):
        # TODO maybe move to child-class due to specific form (e.g. .dat)
        """ This should be a docstring """
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
                postprocessing_list = [self.postprocessor, output_file_opt, option, target_file_opt]
                postprocess_command = ' '.join(filter(None, postprocessing_list))
                _, _, _ = self.run_subprocess(postprocess_command)
        else:
            target_file_opt = os.path.join('--output=' + target_file_base_name, self.file_prefix)
            postprocessing_list = [self.postprocessor, output_file_opt, target_file_opt]
            postprocess_command = ' '.join(filter(None, postprocessing_list))
            _, _, _ = self.run_subprocess(postprocess_command)

    def do_postpostprocessing(self):
        """ Run script to extract results from monitor file

        Args:

        Returns:
            float: actual simulation result
            Assemble post processing command """
        dest_dir = os.path.join(str(self.experiment_dir), str(self.job_id))
        output_directory = os.path.join(dest_dir, 'output')
        if self.job['status'] != "failed":
            # this is a security duplicate in case post_post did not catch an error
            self.result = None
            self.result = self.postpostprocessor.postpost_main(output_directory)
            sys.stderr.write("Got result %s\n" % (self.result))

    # ---------------- CHILDREN METHODS THAT NEED TO BE IMPLEMENTED ---------------
    @abc.abstractmethod
    def setup_dirs_and_files(self):
        """ this should be a docstring """
        pass

    @abc.abstractmethod
    def run_job(self):
        """ Actual method to run the job on computing machine
            using run_subprocess method from base class
        """
        pass
