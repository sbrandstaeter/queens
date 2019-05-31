from pqueens.drivers.driver import Driver

#### some comments that will be deleted later #####
# so far driver options are assambled in cluster_scheduler file which seems to be wrong
# ssh command is fully designed and then executed the driver file with the main function reading the argument specified in the ssh command after the script name (sys.args[1]) -> similar to argsparser!

#----
# here rather than reading the command: build driver from config (copy partly the stuff done in the scheduler!
# this should be here!
# so far the DB was not used in local baci driver --> this should be changed!
class Baci_driver_native(Driver):
    """ Driver to run BACI natively on workstation

        Args:
            job (dict): Dict containing all information to run the simulation

        Returns:
            float: result
    """
    def __init__(self, base_settings):
        super(Baci_driver_native, self).__init__(base_settings)


    @classmethod
    def from_config_create_driver(cls, config, base_settings):
        """ Create Driver from input file

        Args:
            base_settings: Driver specifications declared in base class (need to be passed through child class and back to __init__ of base class
        Returns:
            driver: Baci_driver_native object

        """

        return cls(base_settings)

    def setup_mpi(self): # TODO this is not needed atm
        pass
