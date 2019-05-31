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
        scheduler = scheduler_class.from_config_create_scheduler(config, base_settings, scheduler_name=None)
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
        return stdout, stderr, p #TODO if poll and return p is helpful


########## Base class methods ###########################################################


######### Children methods that need to be implemented / abstractmethods ######################
    @abc.abstractmethod # how to check this is dependent on cluster / env
    def alive(self,process_id):
        pass

    @abc.abstractmethod
    def submit(self, job_id, batch):
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
