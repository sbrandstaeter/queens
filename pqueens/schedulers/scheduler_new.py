import abc

class Scheduler(metaclass=abc.ABCMeta):
    """ Base class for schedulers """

    def __init__(self, base_settings):
        # add base class scheduler stuff here

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
        from .schedulers import Local_scheduler
        from .schedulers import PBS_scheduler
        from .schedulers import Slurm_scheduler

        scheduler_dict = {'local': LocalScheduler,
                          'pbs': PBSScheduler,
                          'slurm': SlurmScheduler}

        if scheduler_name:
            scheduler_options = config[scheduler_name]
        else:
            scheduler_options = config['scheduler']

        # determine which object to create
        scheduler_class = scheduler_dict[scheduler_options["scheduler_type"]]

########### create base settings #################################################
        scheduler = scheduler_class.from_config_create_scheduler(config, base_settings)
        return scheduler
############Some basic functions that should be added to resources.py and implemented here ############################################################################
    def init():
        pass

    def clean_up():
        pass

########## Auxiliary high-level methods #############################################

    def run_subprocess(command_string, my_env = None):
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
        p.poll()
        print(stderr)
        print(stdout)
        return stdout, p #TODO if poll and return p is helpful

    #TODO: new method that is implemented maybe on this level and takes only care for ssh communication
    # we could make it a class method so that the actual scheduler does not need to be created locally
    # --> we need to check were the actual scheduler gets called/created and change that to a call to this method.
    # --> for local -- either ssh into itself or make an case statement
####################################################################
####################################################################
    def connect_computing_machine(self, command):
        # implement as base method and check if ssh is necessary
        # if not: just execute commands locally
        if self.connect_to_resource in locals(): #TODO: there should be more: file sends and what basic info about next calculation step ...--> this is executed on workstation
            #run ssh command
        run_subprocess(command_string, my_env = None):
        else:
            # run local command

######################################################################
#####################################################################



########## Base class methods ###########################################################
######### Children methods that need to be implemented / abstractmethods ######################
    @abc.abstractmethod # how to check this is dependent on cluster / env
    def alive(self,process_id): #TODO: ok for now; gets called in resources
        pass #TODO is formulated without ssh

    #TODO: method below actually submits the command also on remote but should not be for ssh comm-> gets called locally
     # this is basically the main method --> we should add the construction of the driver class (maybe here)
    @abc.abstractmethod #TODO: This could be main function here!! maybe only a helper function in children class?
    def submit(self, job_id, experiment_name, batch, experiment_dir,
               scheduler_options, database_address):
        pass

########## #TODO Some optional stuff-> prob better done in child-class  #################################################################
    def create_singularity_container(self):
        """ Add current environment to predesigned singularity container for cluster applications """
        pass

    def check_files_remote(self):
        pass

    def send_files_remote(self):
        pass

    def clean_files_remote(self):
        pass
