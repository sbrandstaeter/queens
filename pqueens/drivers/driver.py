import abc
from pqueens.database.mongodb import MongoDB
from pqueens.utils.injector import inject
import sys
import subprocess
import time
import importlib.util

class Driver(metaclass=abc.ABCMeta)
    """ Base class for Drivers

    This Driver class is the base class for drivers that actually execute a job on
    a computing resource. It is supposed to unify the interface of drivers and
    fully integrate them in QUEENS to enable testing. Furthermore, an abstract
    driver class will give rise to the usage of singularity containers for HPC
    applications.

    Attributes:

    """

    def __init__(self, base_settings): # database object passed by scheduler, scheduler has it from
        # add base class driver stuff here
        self.experiment_dir = base_settings['experiment_name']
        self.job_id = base_settings['job_id']
        self.input_file = base_settings['input_file']
        self.output_file = base_settings['output_file']
        self.job = base_settings['job']
        self.batch = base_settings['batch']
        self.executable = base_settings['executable']
        self.database = base_settings['databank']
        self.result = base_settings['result']
        self.postprocessor = base_settings['postprocessor']
        self.post_options = base_settings['post_options']
        self.postpostprocessor = base_settings['postpostprocessor']
        self.mpi_config = None # config mpi for each machine (command itself is organized via scheduler)

    @classmethod
    def from_config_create_driver(cls, config, driver_type=None, database=None, scheduler_obj = None):
        """ Create driver from problem description

        Args:
            config (dict):      Dictionary with QUEENS problem description
            driver_type (str):  Name of driver to identify right section in options
                                dict (optional)
            database (database):database to use (optional)

        Returns:
            driver: Driver object

        """
        from .drivers import Ansys_driver_native
        from .drivers import Baci_driver_bruteforce
        from .drivers import Baci_driver_docker #TODO: Check if this can be deleted toghether with gen_driver -> seems like needed only for tests? Check with JB
        from .drivers import Baci_driver_native
        from .drivers import Baci_driver_kaiser #TODO: Rename this driver file
        from .drivers import Fenics_driver_native #TODO: Rename this driver file and check it
        from .drivers import Fenics_driver_bruteforce
        from .drivers import Python_driver

        driver_dict = {'ansys_native': Ansys_driver_native,
                       'baci_bruteforce': Baci_driver_bruteforce,
                       'baci_docker': Baci_driver_docker,
                       'baci_native': Baci_driver_native, # TODO: not necessary..
                       'baci_kaiser': Baci_driver_kaiser,
                       'fenics_native': Fenics_driver_native,
                       'fenics_bruteforce': Fenics_driver_bruteforce}

        if driver_type is None:
            driver_version = config['driver']['driver_type']
            driver_class = driver__dict[driver_version]
        if scheduler_obj:
            self.mpi_config = {'num_procs' = scheduler_obj.num_procs_per_node}

 ###### create base settings #####################
            driver_options = config['driver']['driver_options']
            base_settings['experiment_name']= driver_options['experiment_dir']
            base_settings['job_id']= driver_options['job_id']
            base_settings['input_file']=None
            base_settings['output_file']=None
            base_settings['job']= None
            base_settings['batch']=driver_options['batch']
            base_settings['executable']=driver_options['path_to_executable']
            base_settings['databank']=MongoDB(database_address=driver_options['database_address'])
            base_settings['result']=None
            base_settings['postprocessor']=driver_options['path_to_postprocessor']
            base_settings['post_options']=driver_options['post_process_options']
            base_settings['postpostprocessor']=driver_options['post_post_script']


            driver = driver_class.from_config_create_driver(config, base_settings)
        else:

           sys.stderr.write("No driver <driver_type> is given in JSON file! Please specify a <driver_type>")
           # driver_version = config[driver_name]['driver_name']
           # driver_class = driver_dict[driver_version]
           # driver = driver_class.from_config_create_driver(config, driver_name, database, base_settings) # check arguments -> should be args of childclass

        return driver

    def main_run(self):
        """ Actual main method of the driver """
        self.prepare_environment()
        self.init_job()
        self.run_job()
        self.finish_job()

##### Auxiliary high-level methods ########################################

    def prepare_environment(self):
        """ Prepare the environment for computing """

        self.setup_dirs_and_files()
        self.setup_mpi()

    def finish_and_clean(self):
        """ Finish and clean the resources and environment """

        self.finish_job()
        self.do_postprocessing()
        self.do_postpostprocessing() # can be passed in child class

    def run_subprocess(command_string, my_env = None)
        """ Method to run command_string outside of Python """
        if (my_env is None) and ('my_env' in mpi_config):
            p = subprocess.Popen(command_string,
                             env = self.mpi_config['my_env'],
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             shell=True,
                             universal_newlines=True)
        else:
            p = subprocess.Popen(command_string,
                             env = my_env,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             shell=True,
                             universal_newlines=True)

        stdout, stderr = p.communicate()
        print(stderr)
        print(stdout)
        return stdout
##### Base class methods ##################################################

    def setup_dirs_and_files(self):
        """ Setup directory structure

            Args:
                driver_options (dict): Options dictionary

            Returns:
                str, str, str: simualtion prefix, name of input file, name of output file
        """
        # base directories
        dest_dir = str(self.experiment_dir) + '/' + \
                  str(self.job_id)

        prefix = str(self.experiment_name) + '_' + \
                 str(self.job_id)

        # Depending on the input file, directories will be created locally or on a cluster
        output_directory = os.path.join(dest_dir, 'output')
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        # create input file name
        self.input_file = dest_dir + '/' + str(self.experiment_name) + \
                          '_' + str(self.job_id) + '.dat'

        # create output file name
        self.output_file =  output_directory + '/' + str(self.experiment_name) + \
                          '_' + str(self.job_id)


    def init_job(self):
        """ Initialize job in database


            Returns:
                dict: Dictionary with job information

        """
        # Create database object and load the already initiated job entry
        self.job = self.database.load(self.experiment_name, self.batch, 'jobs', {'id' : self.job_id})

        # start settings for job
        self.job['start time'] = time.time()

        # save the job with the new start time
        self.database.save(self.job, self.experiment_name, 'jobs', self.batch,
                {'id' : self.job_id})

        sys.stderr.write("Job launching after %0.2f seconds in submission.\n"
                         % (start_time-self.job['submit time']))


    def run_job(self):
        """ Actual method to run the job on computing machine
            using run_subprocess method from base class
        """

        # assemble run command
        command_list = [self.mpi_config['mpi_run'], self.mpi_config['flags'], self.executable, self.input_file, self.output_file]
        command_string = ' '.join(command_list)
        _ = self.run_subprocess(command_string)

    def finsih_job(self):
        """ Change status of job to compleded in database """

        if self.result is None:
            #TODO: errormsg
        else:
            self.job['result'] = self.result #TODO: Throw error for result = None
            self.job['status'] = 'complete'
            self.job['end time'] = time.time()
            self.database.save(self.job, self.experiment_name, 'jobs', self.batch, {'id' : self.job_id})


    def do_postprocessing(self):
        #TODO: Check if this is abstract enough --> --file could be troublesome
        output_file_opt = '--file=' + self.output_file
        postprocessing_list = [self.mpi_config['mpi_run'], '-np 1', self.postprocessor, output_file_opt, self.post_options]
        postprocess_command = ' '.join(postprocessing_list)
        _ = self.run_subprocess(postprocess_command)


    def do_postpostprocessing(self):
        """ Run script to extract results from monitor file

        Args:
            post_post_script (string): name of script to run
            baci_output_file (string): name of file to use

        Returns:
            float: actual simulation result
        """
        """ Assemble post processing command """

        result = None
        if self.postpostprocessor != None:
            spec = importlib.util.spec_from_file_location("module.name", self.postpostprocessor)
            post_post_proc = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(post_post_proc)
            result, error = post_post_proc.run(baci_output)
        else:
            raise RuntimeError("You need to provide post_post_script")

        # cleanup of unnecessary data after QoI got extracted and flag is set in config
        if self.post_options["delete_field_data"].lower()=="true":
            # Delete every ouput file exept the .mon file
            # --> use baci_output to get path to current folder
            # --> start subprocess to delete files with linux commands
            command_string = "cd "+ baci_output + "&& ls | grep -v *.mon | xargs rm" # This is the actual linux commmand
            _ = self.run_subprocess(command_string)

        # Put files that were not compliant with the requirements from the
        # postpost_processing scripts in a special folder and do not pass on result
        # of those files
        if error !="":
            result = None
            command_string = "cd "+ baci_output + "&& cd ../.. && mkdir -p postpost_error && cd " + baci_ouput + "&& cd .. && mv *.dat ../postpost_error/" # This is the actual linux commmand
            _ = self.run_subprocess(command_string)

        self.result = result
        sys.stderr.write("Got result %s\n" % (self.result))

##### Children methods that need to be implemented #########################

    @abc.abstractmethod
    def setup_mpi(self):
        """ Configure and set up the environment for multi_threats """
        pass

#### TODO Optional methods ##########################################################

    def create_singularity_container(self):
        """ Add current environment to predisigned singularity container
            for cluster executions """
        pass
