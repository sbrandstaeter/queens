from pqueens.drivers.driver import Driver
from pqueens.database.mongodb import MongoDB

class Baci_driver_native(Driver):
    """ Driver to run BACI natively on workstation

        Args:
            job (dict): Dict containing all information to run the simulation

        Returns:
            float: result
    """
    def __init__(self, base_settings):
        super(Baci_driver_native, self).__init__(base_settings)
        self.mpi_config={}
        address ='localhost:27017'
        self.database = MongoDB(database_address=address)



    @classmethod
    def from_config_create_driver(cls, config, base_settings):
        """ Create Driver from input file

        Args:
            base_settings: Driver specifications declared in base class (need to be passed through child class and back to __init__ of base class
        Returns:
            driver: Baci_driver_native object

        """
        base_settings['experiment_name']=config['global_settings']['experiment_name']
        return cls(base_settings)

    def setup_mpi(self,num_procs): # TODO this is not needed atm
        pass
