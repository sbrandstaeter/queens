"""Job interface class."""

from pqueens.drivers import from_config_create_driver
from pqueens.interfaces.interface import Interface
from pqueens.schedulers import from_config_create_scheduler


class JobInterface(Interface):
    """Class for mapping input variables to responses.

    Attributes:
        scheduler (Scheduler):      scheduler for the simulations
        driver (Driver):            driver for the simulations
    """

    def __init__(
        self,
        scheduler,
        driver,
    ):
        """Create JobInterface.

        Args:
            scheduler (Scheduler):      scheduler for the simulations
            driver (Driver):            driver for the simulations
        """
        super().__init__()
        self.scheduler = scheduler
        self.driver = driver
        self.scheduler.copy_file(self.driver.simulation_input_template)

    @classmethod
    def from_config_create_interface(cls, interface_name, config):
        """Create JobInterface from config dictionary.

        Args:
            interface_name (str):   Name of interface.
            config (dict):          Dictionary containing problem description.

        Returns:
            interface: Instance of JobInterface
        """
        interface_options = config[interface_name]

        # create driver from config
        driver = from_config_create_driver(
            driver_name=interface_options["driver_name"],
            config=config,
        )

        # create scheduler from config
        scheduler = from_config_create_scheduler(
            scheduler_name=interface_options["scheduler_name"],
            config=config,
        )

        # instantiate object
        return cls(scheduler=scheduler, driver=driver)

    def evaluate(self, samples):
        """Evaluate.

        Args:
            samples (np.array): Samples of simulation input variables

        Returns:
            output (dict): Output data
        """
        samples_list = self.create_samples_list(samples)
        output = self.scheduler.evaluate(samples_list, driver=self.driver)

        return output
