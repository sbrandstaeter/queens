import abc

class Scheduler(metaclass=abc.ABCMeta):
    """ Base class for schedulers """

    def __init__(self): #TODO: this is new so check for correct parsing

    @classmethod
    def from_config_create_scheduler(cls, scheduler_name, config):
        """ Create scheduler from problem description

        Args:
            scheduler_name (string): Name of scheduler
            config (dict): Dictionary with QUEENS problem description

        Returns:
            scheduler: scheduler object

        """
        # import here to avoid issues with circular inclusion
        import pqueens.schedulers.local_scheduler
        import pqueens.schedulers.PBS_scheduler
        import pqueens.schedulers.slurm_scheduler

        scheduler_dict = {'local': pqueens.schedulers.local_scheduler.LocalScheduler,
                          'pbs': pqueens.schedulers.PBS_scheduler.PBSScheduler,
                          'slurm': pqueens.schedulers.slurm_scheduler.SlurmScheduler}

        scheduler_options = config[scheduler_name]
        # determine which object to create
        scheduler_class = scheduler_dict[scheduler_options["scheduler_type"]]

        return scheduler_class.from_config_create_scheduler(scheduler_name,
                                                            config)

##################################### executed locally e.g. remote ######################################
    #TODO: method below actually submits the command also on remote but should not be for ssh comm-> gets called locally
     # this is basically the main method --> we should add the construction of the driver class (maybe here)
    @abc.abstractmethod
    def submit(self, job_id, experiment_name, batch, experiment_dir,
               scheduler_options, database_address):
        pass

################################# executed only on workstation #####################################
    #TODO: new method that is implemented maybe on this level and takes only care for ssh communication
    # we could make it a class method so that the actual scheduler does not need to be created locally
    # --> we need to check were the actual scheduler gets called/created and change that to a call to this method.
    # --> for local -- either ssh into itself or make an case statement

    def connect_ssh()
###################################################################################################

    @abc.abstractmethod
    def alive(self,process_id):
        pass
