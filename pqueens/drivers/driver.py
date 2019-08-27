import abc
from pqueens.database.mongodb import MongoDB
from pqueens.utils.injector import inject
import sys
import subprocess
import time
import os
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
        self.experiment_dir = base_settings['experiment_dir']
        self.experiment_name = base_settings['experiment_name']
        self.job_id = base_settings['job_id']
        self.input_file = base_settings['input_file']
        self.template = base_settings['template']
        self.output_file = base_settings['output_file']
        self.job = base_settings['job']
        self.batch = base_settings['batch']
        self.executable = base_settings['executable']
        self.result = base_settings['result']
        self.postprocessor = base_settings['postprocessor']
        self.post_options = base_settings['post_options']
        self.postpostprocessor = base_settings['postpostprocessor']
        self.mpi_flags = {} # config mpi for each machine (build on runtime)
        self.pid=None
        self.port=base_settings['port']
        self.num_procs = base_settings['num_procs']
        self.num_procs_post = base_settings['num_procs_post']

    @classmethod
    def from_config_create_driver(cls, config, job_id, batch, port=None, abs_path=None):
        """ Create driver from problem description

        Args:
            config (dict):      Dictionary with QUEENS problem description
            driver_type (str):  Name of driver to identify right section in options
                                dict (optional)
            database (database):database to use (optional)

        Returns:
            driver: Driver object

        """
        from pqueens.drivers.baci_driver_bruteforce import Baci_driver_bruteforce
        from pqueens.drivers.baci_driver_native import Baci_driver_native
        from pqueens.drivers.baci_driver_schmarrn import Baci_driver_schmarrn
        if abs_path== None:
            from pqueens.post_post.post_post import Post_post
        else:
            print(abs_path)
            import importlib.util
            spec = importlib.util.spec_from_file_location("post_post", abs_path)
            post_post = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(post_post)
            try:
                from post_post.post_post import Post_post
            except:
                print('fail')
                raise ImportError('Could not import the post_post module!')


        driver_dict = {'baci_bruteforce': Baci_driver_bruteforce,
                       'baci_native': Baci_driver_native,
                       'baci_schmarrn': Baci_driver_schmarrn}
        driver_version = config['driver']['driver_type']
        driver_class = driver_dict[driver_version]
###### create base settings #####################
        driver_options = config['driver']['driver_params']
        first = list(config['resources'])[0]
        scheduler_name = config['resources'][first]['scheduler']
        base_settings = {}
        base_settings['num_procs'] = config[scheduler_name]['num_procs']
        base_settings['num_procs_post'] = config[scheduler_name]['num_procs_post']
        base_settings['experiment_dir']= driver_options['experiment_dir']
        base_settings['job_id']= job_id
        base_settings['input_file']=None
        base_settings['template']=driver_options['input_template']
        base_settings['output_file']=None
        base_settings['job']= None
        base_settings['batch']=batch
        base_settings['executable']=driver_options['path_to_executable']
        base_settings['result']=None
        base_settings['port']=port
        base_settings['postprocessor']=driver_options['path_to_postprocessor']
        base_settings['post_options']=driver_options['post_process_options']
        base_settings['postpostprocessor']= Post_post.from_config_create_post_post(config)
        driver = driver_class.from_config_create_driver(config, base_settings)

        return driver

    def main_run(self):
        """ Actual main method of the driver """
        self.prepare_environment()
        self.init_job()
        self.run_job()
        self.finish_and_clean()

##### Auxiliary high-level methods ########################################

    def prepare_environment(self):
        """ Prepare the environment for computing """
        self.setup_dirs_and_files()

    def finish_and_clean(self):
        """ Finish and clean the resources and environment """
        self.do_postprocessing()
        self.do_postpostprocessing()
        self.finish_job()


    def run_subprocess(self,command_string, my_env = None):
        """ Method to run command_string outside of Python """
        p = subprocess.Popen(command_string,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         shell=True,
                         universal_newlines=True)

        stdout, stderr = p.communicate()
        process_id = p.pid
        return stdout,stderr,process_id
##### Base class methods ##################################################

    def setup_dirs_and_files(self): #TODO: Dat is hard coded! Change this!
        """ Setup directory structure

            Args:
                driver_options (dict): Options dictionary

            Returns:
                str, str, str: simualtion prefix, name of input file, name of output file
        """
        # base directories
        dest_dir = str(self.experiment_dir) + '/' + \
                  str(self.job_id)

        prefix = str(self.experiment_dir) + '_' + \
                 str(self.job_id)

        # Depending on the input file, directories will be created locally or on a cluster
        output_directory = os.path.join(dest_dir, 'output')
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        # create input file name
        self.input_file = dest_dir + '/' + str(self.experiment_name) + \
                          '_' + str(self.job_id) + '.dat' #TODO change hard coding of .dat

        # create output file name
        self.output_file =  output_directory + '/' + str(self.experiment_name) + \
                          '_' + str(self.job_id)


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
        # save the job with the new start time
        self.database.save(self.job, self.experiment_dir, 'jobs', self.batch,
                {'id' : self.job_id})

        sys.stderr.write("Job launching after %0.2f seconds in submission.\n"
                         % (start_time-self.job['submit time']))

        # create actual input file with parsed parameters
        inject(self.job['params'],self.template,self.input_file)

    def run_job(self):
        """ Actual method to run the job on computing machine
            using run_subprocess method from base class
        """
        # assemble run command
        self.setup_mpi(self.num_procs)
        command_list = ['mpirun', '-np', str(self.num_procs), self.mpi_flags, self.executable, self.input_file, self.output_file]
        command_string = ' '.join(filter(None,command_list))
        stdout,stderr,self.pid = self.run_subprocess(command_string)
        if stderr !="":
            raise RuntimeError(stderr+stdout)

    def finish_job(self):
        """ Change status of job to compleded in database """

        if self.result is None:
            raise RuntimeError("No result!")
        else:
            self.job['result'] = self.result
            self.job['status'] = 'complete'
            self.job['end time'] = time.time()
            self.database.save(self.job, self.experiment_name, 'jobs', str(self.batch), {'id' : self.job_id})


    def do_postprocessing(self):
        #TODO: Check if this is abstract enough --> --file could be troublesome
        self.setup_mpi(self.num_procs_post)
        target_file_base_name = os.path.dirname(self.output_file)
        output_file_opt = '--file=' + self.output_file
        for num,option in enumerate(self.post_options):
            target_file_opt = '--output=' + target_file_base_name + "/QoI_" + str(num+1)
            postprocessing_list = ['mpirun', '-np', str(self.num_procs_post), self.mpi_flags, self.postprocessor, output_file_opt, option, target_file_opt] #TODO: number of procs for drt_monitor must be one but we should provide an options to control the procs for other post_processors
            postprocess_command = ' '.join(filter(None,postprocessing_list))
            _,stderr,_ = self.run_subprocess(postprocess_command)


    def do_postpostprocessing(self): #TODO: file extentions are hard coded we need to change that!
        """ Run script to extract results from monitor file

        Args:

        Returns:
            float: actual simulation result
        """
        """ Assemble post processing command """
        result = None
        result, error = self.postpostprocessor.read_post_files(self.output_file)
        # cleanup of unnecessary data after QoI got extracted and flag is set in config
        if self.postpostprocessor.delete_field_data.lower()=="true":
            # Delete every ouput file exept the .mon file
            # --> use self.output to get path to current folder
            # --> start subprocess to delete files with linux commands
            command_string = "cd "+ self.output_file + "&& ls | grep -v --include=*.{mon,csv} | xargs rm" # TODO check if this works for several extentions
            _,stderr,_ = self.run_subprocess(command_string) #TODO catch posbl. errors

        # Put files that were not compliant with the requirements from the
        # postpost_processing scripts in a special folder and do not pass on result
        # of those files
        if error == 'true':
            result = None
            command_string = "cd "+ self.output_file + "&& cd ../.. && mkdir -p postpost_error && cd " + self.output_file + "&& cd .. && mv *.dat ../postpost_error/" # This is the actual linux commmand
            _,stderr,_ = self.run_subprocess(command_string)

        self.result = result
        sys.stderr.write("Got result %s\n" % (self.result))

##### Children methods that need to be implemented #########################

    @abc.abstractmethod
    def setup_mpi(self,ntasks):
        """ Configure and set up the environment for multi_threats """
        pass
