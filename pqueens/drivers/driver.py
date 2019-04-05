import abc
from pqueens.database.mongodb import MongoDB
from pqueens.utils.injector import inject

class Driver(metaclass=abc.ABCMeta)
    """ Base class for Drivers

    This Driver class is the base class for drivers that actually execute a job on
    a computing resource. It is supposed to unify the interface of drivers and
    fully integrate them in QUEENS to enable testing. Furthermore, an abstract
    driver class will give rise to the usage of singularity containers for HPC
    applications.

    Attributes:

    """

    def __init__(self, global_settings): # database object passed by scheduler, scheduler has it from
        self.database = MongoDB(database_address=global_settings['db_adress'])
        self.global_settings = global_settings

    @classmethod
    def from_config_create_driver(cls, config, driver_name=None, database=None):
        """ Create driver from problem description

        Args:
            config (dict):      Dictionary with QUEENS problem description
            driver_name (str):  Name of driver to identify right section in options
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
                       'baci_native': Baci_driver_native,
                       'baci_kaiser': Baci_driver_kaiser,
                       'fenics_native': Fenics_driver_native,
                       'fenics_bruteforce': Fenics_driver_bruteforce}

        if driver_name is None:
            driver_version = config['driver']['driver_name']
            driver_class = driver__dict[driver_version]
            driver = driver_class.from_config_create_driver(config, database)
        else:
            driver_version = config[driver_name]['driver_name']
            driver_class = driver_dict[diver_version]
            driver = driver_class.from_config_create_driver(config, driver_name, database)

        return driver

    def main_run(self):
        """ Actual main method of the driver """
        self.prepare_environment()
        self.init_job()
        self.run_job()
        self.finish_job()

##### Auxiliary high level methods ########################################

    def prepare_environment(self):
        """ Prepare the environment for computing """

        self.setup_dirs_and_files()
        self.setup_mpi()

    def finish_and_clean(self):
        """ Finish and clean the resources and environment """

        self.finish_job()
        self.do_postprocessing()
        self.do_postpostprocessing() # can be passed in child class


##### Children methods that need to be implemented #########################

    @abc.abstractmethod
    def setup_dirs_and_files(self):
        """ Setup directory structure """
        pass

    @abc.abstractmethod
    def init_job(self):
        """ Initialize job in database """
        pass

    @abc.abstractmethod
    def run_job(self):
        """ Actual method to run the job on computing machine """
        pass

    @abc.abstractmethod
    def finsih_job(self):
        """ Change status of job to compleded in database """
        pass

    @abc.abstractmethod
    def do_postprocessing(self):
        """ Assemble post processing command """
       pass

    @abc.abstractmethod
    def do_postpostprocessing(self):
        """ Assemble post post processing command """
       pass

    @abc.abstractmethod
    def setup_mpi(self):
        """ Configure and set up the environment for multi_threats """
        pass

#### Optional methods ##########################################################


    def get_num_nodes(self):
        """ determine number of processers from nodefile """
        pass

    def create_singularity_container(self):
        """ Add current environment to predisigned singularity container
            for cluster executions """
        pass
